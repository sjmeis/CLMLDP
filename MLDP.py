# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.special import lambertw, softmax
import scipy.stats as sct
from scipy.linalg import sqrtm
import nltk
nltk.download('wordnet', quiet=True)
from nltk.corpus import wordnet
import numba as nb
import math
import faiss
import random
import gensim.models
from sklearn.feature_extraction.text import TfidfVectorizer

import importlib_resources as impresources

faiss_num_list = 100
faiss_num_probe = 50

@nb.njit(fastmath=True, parallel=True)
def calc_distance(vec_1,vec_2):
    res = np.empty((vec_1.shape[0],vec_2.shape[0]),dtype=vec_1.dtype)
    for i in nb.prange(vec_1.shape[0]):
        for j in range(vec_2.shape[0]):
            res[i,j] = np.sqrt((vec_1[i,0]-vec_2[j,0])**2+(vec_1[i,1]-vec_2[j,1])**2+(vec_1[i,2]-vec_2[j,2])**2)
    return res

def calc_probability(embed1, embed2, epsilon=2):
    distance = calc_distance(embed1, embed2)
    sim_matrix = -distance
    prob_matrix = softmax(epsilon * sim_matrix / 2, axis=1)
    return prob_matrix

def euclidean_dt(m, v):
    diff = m - v
    dist = np.sum(np.square(diff), axis=-1)
    return np.sqrt(dist)

def truncated_Poisson(mu, max_value, size):
    temp_size = size
    while True:
        temp_size *= 2
        temp = sct.poisson.rvs(mu, size=temp_size)
        truncated = temp[temp <= max_value]
        if len(truncated) >= size:
            return truncated[:size]

def truncated_Gumbel(mu, scale, max_value, size):
    temp_size = size
    while True:
        temp_size *= 2
        temp = np.random.gumbel(loc = mu, size=temp_size, scale = scale.real)
        truncated = temp[np.absolute(temp) <= max_value]
        if len(truncated) >= size:
            return truncated[:size]
        
def euclidean_laplace_rand_fn(dimensions, epsilon):
    v = np.random.multivariate_normal(mean = np.zeros(dimensions),
                                        cov = np.eye(dimensions))
    v_norm = np.linalg.norm(v) + 1e-30
    v = v / v_norm

    l = np.random.gamma(shape = dimensions, scale = 1 / epsilon)
    return l * v


class MultivariateCalibrated:
    def __init__(self, embed=None,
                 embedding_matrix=None, 
                 dim=300, 
                 use_faiss=True, 
                 return_noise=False):
        if embedding_matrix is not None:
            self.embedding_matrix = embedding_matrix
        else:
            self.embedding_matrix = gensim.models.KeyedVectors.load_word2vec_format(embed, binary=False, unicode_errors="ignore")
        self.dim = dim
        self.use_faiss = use_faiss
        self.return_noise = return_noise

        if use_faiss:
            nlist = faiss_num_list
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            self.index.train(self.embedding_matrix.vectors)
            self.index.add(self.embedding_matrix.vectors)
            self.index.probe = faiss_num_probe

    def get_perturbed_vector(self, word_vec, epsilon):
        noise = np.random.multivariate_normal(np.zeros(self.dim), np.identity(self.dim))
        norm_noise = noise / np.linalg.norm(noise)
        N = np.random.gamma(self.dim, 1/epsilon) * norm_noise
        return word_vec + N, N

    def get_nearest(self, vector):
        if self.use_faiss:
            _, I = self.index.search(np.array([vector.astype('float32')]), k=1)
            return I[0][0]
        else:
            diff = (self.embedding_matrix.vectors - vector)
            most_sim_index = np.argmin(np.linalg.norm(diff, axis=1))
            return most_sim_index

    def replace_word(self, word, epsilon):
        N = None
        if word in self.embedding_matrix:
            embedding_vector = self.embedding_matrix[word]
        else:
            embedding_vector = None

        if embedding_vector is not None:
            perturbed_vector, noise = self.get_perturbed_vector(embedding_vector, epsilon)
            sim_ind = self.get_nearest(perturbed_vector)
            new_word = self.embedding_matrix.index_to_key[sim_ind]
            if self.return_noise == True:
                if new_word is None:
                    return word, noise
                else: 
                    return new_word, noise
            else:
                if new_word is None:
                    return word
                else: 
                    return new_word
        if self.return_noise == True:
            return word, None
        else:
            return word

class VickreyMechanism:
    def __init__(self, embed=None,
                 embedding_matrix=None, 
                 dim=300,
                 k = 2, t = [1, 0],
                 use_faiss=True,
                 return_noise=False):
        if embedding_matrix is not None:
            self.embedding_matrix = embedding_matrix
        else:
            self.embedding_matrix = gensim.models.KeyedVectors.load_word2vec_format(embed, binary=False, unicode_errors="ignore")
        self.dim = dim
        self.num_perturbed = 0
        self.num_words = 0
        self.k = k
        self.t = np.asarray(t)
        self.use_faiss = use_faiss

        self.return_noise = return_noise

        if use_faiss:
            nlist = faiss_num_list
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            self.index.train(self.embedding_matrix.vectors)
            self.index.add(self.embedding_matrix.vectors)
            self.index.probe = faiss_num_probe
    
    def replace_word(self, word, epsilon):    
        if word in self.embedding_matrix:
            word_embed = self.embedding_matrix[word]
        else:
            word_embed = None

        if word_embed is None: 
            return word

        noise = euclidean_laplace_rand_fn(dimensions=self.dim, epsilon=epsilon)
        noisy_vector = word_embed + noise

        if self.use_faiss:
            D, I = self.index.search(np.array([noisy_vector.astype('float32')]), int(self.k + 1))
            indices = I[0]
            dists = D[0]
        else:
            dists = euclidean_dt(self.embedding_matrix.vectors, noisy_vector)
            indices = np.argsort(dists)
            dists = dists[indices]

        if indices[0] == self.embedding_matrix.key_to_index[word]:
            idx = indices[1:self.k + 1]
            dist = dists[1: self.k + 1]
        else:
            idx = indices[:self.k]
            dist = dists[:self.k]

        p = -self.t * dist
        p = softmax(p)

        i = np.random.choice(idx, p = p)

        perturbed_word = self.embedding_matrix.index_to_key[i]
        if self.return_noise == True:
            if perturbed_word is None:
                return word, noise
            else:
                return perturbed_word, noise
        else:
            if perturbed_word is None:
                return word
            else:
                return perturbed_word

class TEM:
    def __init__(self, embed=None,
                embedding_matrix=None, 
                dim=300,
                use_faiss=True,
                return_noise=False):
        if embedding_matrix is not None:
            self.embedding_matrix = embedding_matrix
        else:
            self.embedding_matrix = gensim.models.KeyedVectors.load_word2vec_format(embed, binary=False, unicode_errors="ignore")
        self.vocab_size = self.embedding_matrix.vectors.shape[0]
        self.dim = dim
        self.return_noise = return_noise

    def replace_word(self, input_word, epsilon, threshold=0.5):     
        if input_word in self.embedding_matrix:
            word_embed = self.embedding_matrix[input_word]
        else:
            word_embed = None

        if word_embed is None: 
            return input_word

        euclid_dists = np.linalg.norm(self.embedding_matrix.vectors - word_embed, axis=1)

        word_euclid_dict = {word:dist for word, dist in zip(self.embedding_matrix.key_to_index.keys(), euclid_dists)}

        beta = 0.001
        
        threshold = round(2/epsilon * math.log(((1-beta)*len(self.embedding_matrix))/beta), 1)

        Lw = [word for word in word_euclid_dict if word_euclid_dict[word] <= threshold]

        f = {word: -word_euclid_dict[word] for word in Lw}

        f["⊥"] = -threshold + 2 * np.log(self.vocab_size/len(Lw)) / epsilon

        noise = [np.random.gumbel(0, 2 / epsilon) for _ in f]
        f = {word: f[word] + noise[i] for i, word in enumerate(f)}
        
        privatized_word = max(f, key=f.get)

        if self.return_noise == True:
            if privatized_word == "⊥":
                new_word = np.random.choice([word for word in self.embedding_matrix.key_to_index.keys() if word not in Lw])
                return new_word, noise
            else:
                return privatized_word, noise
        else:
            if privatized_word == "⊥":
                new_word = np.random.choice([word for word in self.embedding_matrix.key_to_index.keys() if word not in Lw])
                return new_word
            else:
                return privatized_word
            
class Mahalanobis:
    def __init__(self, embed=None,
                embedding_matrix=None, 
                lambd=0.2, 
                dim=300, 
                use_faiss=True,
                return_noise=False):
        self.lambd = lambd
        self.dim = dim
        if embedding_matrix is not None:
            self.embedding_matrix = embedding_matrix
        else:
            self.embedding_matrix = gensim.models.KeyedVectors.load_word2vec_format(embed, binary=False, unicode_errors="ignore")
        self.cov_mat = np.cov(self.embedding_matrix.vectors, rowvar=False) / np.var(self.embedding_matrix.vectors)
        self.identity_mat = np.identity(self.dim)
        self.use_faiss = use_faiss

        self.return_noise = return_noise

        if use_faiss:
            nlist = faiss_num_list
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            self.index.train(self.embedding_matrix.vectors)
            self.index.add(self.embedding_matrix.vectors)
            self.index.probe = faiss_num_probe

    def get_perturbed_vector(self, word_vec, epsilon):
        noise = np.random.multivariate_normal(np.zeros(self.dim), np.identity(self.dim))
        norm_noise = np.divide(noise, np.linalg.norm(noise))
        Z = np.multiply(np.random.gamma(self.dim, 1/epsilon), np.dot(sqrtm(self.lambd*self.cov_mat + (1-self.lambd)*self.identity_mat), norm_noise))
        return word_vec + Z, Z

    def get_nearest(self, vector):
        if self.use_faiss:
            _, I = self.index.search(np.array([vector.astype('float32')]), k=1)
            return I[0][0]
        else:
            diff = (self.embedding_matrix.vectors - vector)
            most_sim_index = np.argmin(np.linalg.norm(diff, axis=1))

            return most_sim_index

    def replace_word(self, word, epsilon):
        if word in self.embedding_matrix:
            embedding_vector = self.embedding_matrix[word]
        else:
            embedding_vector = None

        if embedding_vector is not None:
            perturbed_vector, noise = self.get_perturbed_vector(embedding_vector, epsilon)
            sim_ind = self.get_nearest(perturbed_vector)
            new_word = self.embedding_matrix.index_to_key[sim_ind]
            if self.return_noise == True:
                if new_word is None:
                    return word, noise
                else:
                    return new_word, noise
            else:
                if new_word is None:
                    return word
                else:
                    return new_word
        if self.return_noise == True:
            return word, None
        else:
            return word