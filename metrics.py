from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances


def sent2sentDist(first : List[np.array], second : List[np.array], coef=0):
    """
    Compute distance between two sentances

    Args:
        first: list of embeddings for the first sentance
        second: list of embeddings for the second sentance
        coef: penalty coefficient for the length difference

    Returns:
        final_distance: distance by Mr. Gromov 
    """
    n = len(first)
    m = len(second)
    distances = cosine_distances(first, second)
    indxs = np.unravel_index(np.argsort(distances, axis=None), distances.shape)
    used_first = set()
    used_second = set()
    final_distance = coef * abs(n - m)
    for f, s in zip(indxs[0], indxs[1]):
        if f in used_first or s in used_second:
            continue
        final_distance += distances[f][s]
        used_first.add(f)
        used_second.add(s)
    return final_distance


def doc2docDist(first : List[List[np.array]], second : List[List[np.array]], thold=7, coef=1):
    """
    Compute distance between two documents

    Args:
        first: list of embeddings for the first document
        second: list of embeddings for the second document
        coef: penalty coefficient for the length difference

    Returns:
        final_distance: distance by Mr. Gromov 
    """
    n = len(first)
    m = len(second)
    if n > m:
        first, second = second, first
        n, m = m, n
    res = 0
    for sent1 in first:
        for sent2 in second:
            # print(sent2sentDist(sent1, sent2, coef) )
            if sent2sentDist(sent1, sent2, coef) < thold:
                res += 1
                break
    return res / n