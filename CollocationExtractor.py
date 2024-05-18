import json
import string
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
nltk.download("punkt", quiet=True)
from collections import Counter
import importlib_resources as impresources

def sentence_enum(tokens):
    counts = Counter()
    enum = []
    for t in tokens:
        counts[t] += 1
        enum.append(counts[t])
    return enum

class CollocationExtractor:

    bigrams = None
    trigrams = None
    tokenizer = None

    def __init__(self):
        with open(impresources.files("data") / "bigrams.json", 'r') as f:
            self.bigrams = json.load(f)

        with open(impresources.files("data") / "trigrams.json", 'r') as f:
            self.trigrams = json.load(f)

        self.tokenizer = TweetTokenizer()
        self.detokenizer = TreebankWordDetokenizer()
        self.punct = set(string.punctuation)

    # greedy sequential tokenization (GST)
    def parse(self, text):
        final_tokens = []
        for s in nltk.sent_tokenize(text):
            tokens = [x.lower() for x in self.tokenizer.tokenize(s)]
            enum = sentence_enum(tokens)

            bigram_cands = []
            for i in range(0, len(tokens)-1):
                bigram_cands.append(("{}_{}".format(tokens[i], tokens[i+1]), (enum[i], enum[i+1])))
            bigram_cands = [(x[0], x[1], self.bigrams[x[0]]) if x[0] in self.bigrams and len(x[0].split("_")) == 2 else (x[0], x[1], None) for x in bigram_cands]

            trigram_cands = []
            for i in range(0, len(tokens)-2):
                trigram_cands.append(("{}_{}_{}".format(tokens[i], tokens[i+1], tokens[i+2]), (enum[i], enum[i+1], enum[i+2])))
            trigram_cands = [(x[0], x[1], self.trigrams[x[0]]) if x[0] in self.trigrams and len(x[0].split("_")) == 3 else (x[0], x[1], None) for x in trigram_cands]

            b_candidates = [x for x in bigram_cands if x[2] is not None]
            t_candidates = [x for x in trigram_cands if x[2] is not None]

            top = []
            added = []
            total = 0
            for idx, tup in enumerate(zip(tokens, enum)):
                t = tup[0]
                e = tup[1]
                if (t, e) in added:
                    continue
                t_cands = [x for x in t_candidates if t in x[0].split("_") and e == x[1][x[0].split("_").index(t)]]
                if len(t_cands) == 0:
                    t_cands = [x for x in b_candidates if t in x[0].split("_") and e == x[1][x[0].split("_").index(t)]]

                if len(t_cands) == 0:
                    top.append(t)
                    added.append((t, e))
                else:
                    max_cand = max(t_cands, key=lambda x:x[2])
                    top.append(max_cand[0])
                    total += max_cand[2]
                    temp = []
                    for i, m in enumerate(max_cand[0].split("_")):
                        if idx + i < len(enum):
                            temp.append((m, enum[idx+i]))
                    added.extend(temp)

                    # bi
                    to_del = []
                    for ix, tt in enumerate(max_cand[0].split("_")):
                        tt = tt.lower()
                        to_del.extend([i for i, x in enumerate(b_candidates) if tt in x[0].split("_") and idx+ix < len(enum) and ix < len(x[1]) and enum[idx+ix] == x[1][x[0].split("_").index(tt)]])
                    b_candidates = [x for i, x in enumerate(b_candidates) if i not in to_del]

                    # tri
                    to_del = []
                    for ix, tt in enumerate(max_cand[0].split("_")):
                        tt = tt.lower()
                        to_del.extend([i for i, x in enumerate(t_candidates) if tt in x[0].split("_") and idx+ix < len(enum) and ix < len(x[1]) and enum[idx+ix] == x[1][x[0].split("_").index(tt)]])
                    t_candidates = [x for i, x in enumerate(t_candidates) if i not in to_del]
            final_tokens.append((top, total))

        all_tokens = []
        for x in final_tokens:
            all_tokens.extend(x[0])
        return (all_tokens, sum([x[1] for x in final_tokens]))
    
    # finding best parse from top candidate down (MST)
    def parse_max(self, text):
        final_tokens = []
        for s in nltk.sent_tokenize(text):
            tokens = [x.lower() for x in self.tokenizer.tokenize(s)]
            enum = sentence_enum(tokens)

            bigram_cands = []
            for i in range(0, len(tokens)-1):
                bigram_cands.append(("{}_{}".format(tokens[i], tokens[i+1]), (enum[i], enum[i+1]), i))
            bigram_cands = [(x[0], x[1], self.bigrams[x[0]], x[2]) if x[0] in self.bigrams else (x[0], x[1], None) for x in bigram_cands]

            trigram_cands = []
            for i in range(0, len(tokens)-2):
                trigram_cands.append(("{}_{}_{}".format(tokens[i], tokens[i+1], tokens[i+2]), (enum[i], enum[i+1], enum[i+2]), i))
            trigram_cands = [(x[0], x[1], self.trigrams[x[0]], x[2]) if x[0] in self.trigrams else (x[0], x[1], None) for x in trigram_cands]

            b_candidates = [x for x in bigram_cands if x[2] is not None]
            t_candidates = [x for x in trigram_cands if x[2] is not None]
            candidates = b_candidates + t_candidates + [(x[0], [x[1]], 0, i) for i, x in enumerate(list(zip(tokens, enum)))]
            sorted_candidates = sorted(candidates, key=lambda x: x[2], reverse=True)
            
            total = 0
            added = [(x[0], x[1]) for x in zip(sorted_candidates[0][0].split("_"), sorted_candidates[0][1])]
            top = [(sorted_candidates[0][0], sorted_candidates[0][3])]
            total += sorted_candidates[0][2]
            for i in range(1, len(sorted_candidates)):
                if all((x[0], x[1]) not in added for x in zip(sorted_candidates[i][0].split("_"), sorted_candidates[i][1])):
                    added.extend([(x[0], x[1]) for x in zip(sorted_candidates[i][0].split("_"), sorted_candidates[i][1])])
                    top.extend([(sorted_candidates[i][0], sorted_candidates[i][3])])
                    total += sorted_candidates[i][2]

            combination = ([x[0] for x in sorted(top, key=lambda x: x[1])], total)
            final_tokens.append(combination)

        all_tokens = []
        for x in final_tokens:
            all_tokens.extend(x[0])
        return (all_tokens, sum([x[1] for x in final_tokens]))