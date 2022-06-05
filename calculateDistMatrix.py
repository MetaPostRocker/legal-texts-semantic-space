import numpy as np
from metrics import doc2docDist
import pandas as pd
import pickle
import time
from tqdm import tqdm
from multiprocessing import Pool
import sys, time

emb_dictionary = pickle.load(open("dict_SVD.pkl", 'rb'))
df = pickle.load(open("10000_data.pkl", 'rb'))
corpus = df['clean_motiv_part']

def sentences(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

corpus_sentence = []
for text in corpus:
    corpus_sentence.append([])
    for sentence in list(sentences(text.split(' '), 20))[:30]:
        corpus_sentence[-1].append(' '.join(sentence))
                    
def from_text_to_array(text):
    res = []
    it = 0
    for word in text.split():
        if not word.isalpha():
            continue
        if word in emb_dictionary:
            res.append(emb_dictionary[word])
            it += 1
            if it > 300:
                break
    return res

text_embs = []
for text in tqdm(corpus_sentence):
    lst = list(map(from_text_to_array, text))
    new_lst = list(filter(lambda x: len(x) != 0, lst))
    if len(new_lst) != 0:
        text_embs.append(new_lst)

_, var1, L, R = sys.argv

L = int(L)
R = int(R)
 
def worker(args):
    i, j = args
    return i, j, doc2docDist(text_embs[i], text_embs[j])


args = []
for i in range(L, R + 1):
    for j in range(i, 5000):
        args.append((i, j))


with Pool(10) as p:
      result = list(tqdm(p.imap(worker, args), total=len(args)))

path = 'pairwise_'+var1+'.pkl'
pickle.dump(result, open(path, "wb"))