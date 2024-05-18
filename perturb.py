#!/usr/bin/env python3

import sys
import pandas as pd
from pathlib import Path
import nltk
from gensim.models import KeyedVectors
from tqdm.auto import tqdm
import numpy as np
from nltk.tokenize.treebank import TreebankWordDetokenizer
import json
import time
import string

sys.path.insert(0, "/home/sjmeis/CLMLDP/")

import CollocationExtractor
import MLDP

print("Loading Embeddings")
model = KeyedVectors.load_word2vec_format("/home/sjmeis/embeddings/GoogleNews_300_filtered2.txt", binary=False, unicode_errors="ignore")
phrase = KeyedVectors.load("/home/sjmeis/embeddings/phrase.wordvectors", mmap="r")
phrase_max = KeyedVectors.load("/home/sjmeis/embeddings/phrase_max.wordvectors", mmap="r")

print("Loading Mechanisms")
MECH = "Mahalanobis"
MECH_unigram = MLDP.Mahalanobis(embedding_matrix=model)
MECH_coll = MLDP.Mahalanobis(embedding_matrix=phrase)
MECH_coll_max = MLDP.Mahalanobis(embedding_matrix=phrase_max)

print("Loading Utils")
C = CollocationExtractor.CollocationExtractor()
D = TreebankWordDetokenizer()
PUNCT = set(string.punctuation)

fields = {
    "cola": ["sentence"],
    #"multi": ["premise", "hypothesis"],
    "mrpc": ["sentence1", "sentence2"],
    #"qnli": ["question", "sentence"],
    #"qqp": ["question1", "question2"],
    "rte": ["sentence1", "sentence2"],
    "sst2": ["sentence"],
    #"stsb": ["sentence1", "sentence2"],
    #"wnli": ["sentence1", "sentence2"],
    "yelp": ["review"],
    "trustpilot": ["text"]
}

LOG = "/home/sjmeis/data/privatenlp24/Maha_log.json"
if Path(LOG).is_file() == True:
    with open(LOG, 'r') as f:
        log = json.load(f)
else:
    log = {}

epsilons = [0.1, 1, 5, 10, 25]
SPLIT = ["train", "val"]
for S in SPLIT:
    print(S)
    for file in Path("/home/sjmeis/data/datasets/glue").rglob("*{}.csv".format(S)):
        name = file.stem.split("_")[0]
        if name not in fields:
            continue

        print(file)
        data = pd.read_csv(file)
        if len(data.index) > 10000:
            data = data.sample(n=10000, random_state=42)

        AVG = np.mean([len(nltk.word_tokenize(x.translate(str.maketrans('', '', string.punctuation)))) for x in data[fields[name][0]]])
        for e in epsilons:
            print(e)
            save_name = "{}_{}_{}.csv".format(file.stem, MECH, e)
            if save_name in log:
                continue

            SENTENCE_EPSILON = e * AVG
            replace_word = []
            t1 = 0
            p1 = 0
            tot1 = 0
            replace_phrase_word = []
            t2 = 0
            p2 = 0
            tot2 = 0
            len2 = 0
            replace_phrase = []
            t3 = 0
            p3 = 0
            tot3 = 0
            len3 = 0
            replace_phrase_max = []
            t4 = 0
            p4 = 0
            tot4 = 0
            len4 = 0
            for _, row in tqdm(data.iterrows(), total=len(data.index)):
                text = row[fields[name][0]].lower()
                word_tokens = nltk.word_tokenize(text)
                word_level = SENTENCE_EPSILON / len([x for x in word_tokens if x not in PUNCT])

                start = time.time()
                r = []
                for w in word_tokens:
                    if w in PUNCT:
                        r.append(w)

                    p = MECH_unigram.replace_word(w, epsilon=word_level)
                    if p != w:
                        p1 += 1
                    tot1 += 1
                    r.append(p)
                replace_word.append(D.detokenize(r))
                t1 += time.time() - start

                phrase_tokens = C.parse(text)[0]
                phrase_level = SENTENCE_EPSILON / len([x for x in phrase_tokens if x not in PUNCT])
                
                start = time.time()
                r = []
                for w in phrase_tokens:
                    if w in PUNCT:
                        r.append(w)
                    p = MECH_coll.replace_word(w, epsilon=word_level)
                    if p != w:
                        p2 += 1
                    tot2 += 1
                    r.extend(p.split("_"))
                len2 += len(r) / len(phrase_tokens)
                replace_phrase_word.append(D.detokenize(r))
                t2 += time.time() - start

                start = time.time()
                r = []
                for w in phrase_tokens:
                    if w in PUNCT:
                        r.append(w)
                    p = MECH_coll.replace_word(w, epsilon=phrase_level)
                    if p != w:
                        p3 += 1
                    tot3 += 1
                    r.extend(p.split("_"))
                len3 += len(r) / len(phrase_tokens)
                replace_phrase.append(D.detokenize(r))
                t3 += time.time() - start

                phrase_tokens_max = C.parse_max(text)[0]
                phrase_max_level = SENTENCE_EPSILON / len([x for x in phrase_tokens_max if x not in PUNCT])

                start = time.time()
                r = []
                for w in phrase_tokens_max:
                    if w in PUNCT:
                        r.append(w)
                    p = MECH_coll_max.replace_word(w, epsilon=phrase_max_level)
                    if p != w:
                        p4 += 1
                    tot4 += 1
                    r.extend(p.split("_"))
                len4 += len(r) / len(phrase_tokens_max)
                replace_phrase_max.append(D.detokenize(r))
                t4 += time.time() - start

            data["replace_word"] = replace_word
            data["replace_phrase_word"] = replace_phrase_word
            data["replace_phrase"] = replace_phrase
            data["replace_phrase_max"] = replace_phrase_max
            data.to_csv("/home/ubunsjmeistu/data/privatenlp24/perturbed/{}".format(save_name))

            temp = {}
            temp["t1"] = t1
            temp["pp1"] = p1 / tot1
            temp["t2"] = t2
            temp["pp2"] = p2 / tot2
            temp["len2"] = len2 / len(data.index)
            temp["t3"] = t3
            temp["pp3"] = p3 / tot3
            temp["len3"] = len3 / len(data.index)
            temp["t4"] = t4
            temp["pp4"] = p4 / tot4
            temp["len4"] = len4 / len(data.index)
            print(temp)
            log[save_name] = temp
            with open(LOG, 'w') as out:
                json.dump(log, out, indent=3)
