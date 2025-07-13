import math
from collections import Counter
from typing import *
from typing import Any

import numpy as np
from numpy import floating


def normalize(p: Dict[Hashable, int]) -> Dict[Hashable, float]:
    """
    Normalise a Counter so that the sum of the values is 1.
    :param p: The Counter to normalise
    :return: The normalised Counter
    """
    total = sum(p.values())
    total = total if total > 0 else 1
    return {k: v / total for k, v in p.items()}


def kl_divergence(p: Dict[Hashable, float], q: Dict[Hashable, float]) -> float:
    dkl = 0
    for x in p.keys():
        dkl += p[x] * math.log2(p[x] / q[x])
    return dkl


def average_distribution(p: Dict[Hashable, float], q: Dict[Hashable, float]) -> Dict[Hashable, float]:
    m = dict()
    all_keys = set(p.keys()).union(q.keys())
    for k in all_keys:
        p_k = p.get(k, 0)
        q_k = q.get(k, 0)
        m[k] = (p_k + q_k) / 2
    return m


def jensen_shannon_distance(p: Dict[Hashable, Union[float, int]], q: Dict[Hashable, Union[float, int]],
                            normalized_dict=False, diff_dict=False) -> Union[
    float, Tuple[float, Dict[Hashable, float]]]:
    if not p and not q:
        return 0
    if not p or not q:
        return 1
    if isinstance(p, str):
        p = Counter(p)
    if isinstance(q, str):
        q = Counter(q)
    if not normalized_dict:
        p, q = normalize(p), normalize(q)
    difference_dict = dict() if diff_dict else None
    js_dist = 0
    all_keys = set(p.keys()).union(q.keys())
    for k in all_keys:
        p_k = p.get(k, 0)
        q_k = q.get(k, 0)
        m_k = (p_k + q_k) / 2
        prev = js_dist
        if p_k != 0:
            js_dist += p_k * math.log2(p_k / m_k)
        if q_k != 0:
            js_dist += q_k * math.log2(q_k / m_k)
        if diff_dict:
            difference_dict[k] = js_dist - prev
    if diff_dict:
        difference_dict = normalize(difference_dict)
        return math.sqrt(js_dist * 0.5), difference_dict
    return math.sqrt(js_dist * 0.5)


def analyze_jensen_shannon_distance(data1: List[Dict[Hashable, Union[float, int]]],
                                    data2: List[Dict[Hashable, Union[float, int]]],
                                    top_k: int = 10, verbose=True):
    prototype1, prototype2 = Counter(), Counter()
    for d1, d2 in zip(data1, data2):
        prototype1.update(d1)
        prototype2.update(d2)
    jsd, difference_dict = jensen_shannon_distance(prototype1, prototype2, diff_dict=True)
    if verbose:
        print("Jensen-Shannon distance between data1 and data2: ", jsd)
        sorted_diff = sorted(difference_dict.keys(), key=difference_dict.get, reverse=True)
        print("Main differences:")
        for i in range(top_k):
            print(sorted_diff[i], difference_dict[sorted_diff[i]])
            print(prototype1.get(sorted_diff[i], 0), prototype2.get(sorted_diff[i], 0))
    return jsd, difference_dict


def total_variation_distance(p: Dict[Hashable, Union[float, int]], q: Dict[Hashable, Union[float, int]],
                             normalized_dict=False, diff_dict=False) -> Union[float, Tuple[float, Dict[Hashable, float]]]:
    if not p and not q:
        return 0
    if not p or not q:
        return 1
    if not normalized_dict:
        p, q = normalize(p), normalize(q)
    all_keys = set(p.keys()).union(q.keys())
    difference_dict = dict() if diff_dict else None
    var_dist = 0
    for k in all_keys:
        p_k = p.get(k, 0)
        q_k = q.get(k, 0)
        var_dist += abs(p_k - q_k)
        if diff_dict:
            difference_dict[k] = abs(p_k - q_k)
    if diff_dict:
        difference_dict = normalize(difference_dict)
        return 0.5 * var_dist, difference_dict
    return 0.5 * var_dist


def cosine_similarity(p: Dict[Hashable, Union[float, int]], q: Dict[Hashable, Union[float, int]],
                      normalized_dict=False) -> float:
    if not p and not q:
        return 0
    if not p or not q:
        return 1
    # if not normalized_dict:
    #     p, q = normalize(p), normalize(q)
    cos_sim = 0
    for k in p.keys():
        if k in q:
            cos_sim += p[k] * q[k]
    return cos_sim / (magnitude(p) * magnitude(q))


def magnitude(p: Dict[Hashable, Union[float, int]]) -> float:
    return math.sqrt(sum(x ** 2 for x in p.values()))


def cosine_distance(p: Dict[Hashable, Union[float, int]], q: Dict[Hashable, Union[float, int]],
                    normalized_dict=False) -> float:
    # return 1 - cosine_similarity(p, q, normalized_dict)
    if not p and not q:
        return 0
    if not p or not q:
        return 1
    # if not normalized_dict:
    #     p, q = normalize(p), normalize(q)
    l2_dist = 0
    for k in p.keys():
        if k in q:
            l2_dist += (p[k] - q[k]) ** 2
    return math.sqrt(l2_dist)

def euclidean_distance(p: Dict[Hashable, Union[float, int]], q: Dict[Hashable, Union[float, int]],
                       normalized_dict=False) -> float:
    return math.sqrt(2 * (1 - cosine_similarity(p, q, normalized_dict)))


def cosine_dist_vec(p: np.ndarray, q: np.ndarray) -> float:
    return 1 - (np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q)))


def euclidean_dist_vec(p: np.ndarray, q: np.ndarray) -> float:
    return np.linalg.norm(p - q)
