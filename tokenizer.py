import json
import math
from collections import Counter
from typing import *

import numpy as np

from metrics import normalize


def is_atomic(obj: Any) -> bool:
    return not isinstance(obj, (dict, list))


def _load_json(s: Any) -> Any:
    if isinstance(s, str):
        try:
            return json.loads(s)
        except json.decoder.JSONDecodeError:
            return s
    return s


def tokenize(collection: Union[Dict, List], prefix: str = '', ordered=True,
             mode: Literal["both", "ordered", "unordered", "char"] = "both",
             filter_player=False, binning=False, pair_xy=False) -> Union[List[str], str]:
    tokens = list()
    if mode == "char":
        return str(collection)
    try:
        assert isinstance(collection, (dict, list))
    except AssertionError:
        print(collection)
        print(type(collection))
        # raise AssertionError()
    if isinstance(collection, list):
        for i in range(len(collection)):
            ordered_prefix = prefix + f"[{int(i)}]"
            obj = _load_json(collection[i])
            if is_atomic(obj):
                if ordered and mode != "unordered":
                    tokens.append(ordered_prefix + "." + str(obj))
                if mode != "ordered" or not ordered:
                    tokens.append(prefix + "." + str(obj))
            else:
                if ordered and mode != "unordered":
                    tokens.extend(tokenize(obj, ordered_prefix, mode=mode,
                                           filter_player=filter_player, binning=binning, pair_xy=pair_xy))
                if mode != "ordered" or not ordered:
                    tokens.extend(tokenize(obj, prefix, mode=mode,
                                           filter_player=filter_player, binning=binning, pair_xy=pair_xy))
    elif isinstance(collection, dict):
        # TODO Parameterize this
        if filter_player and "player" in collection.keys() and collection["player"] > 0:  # filter every player but first
            # print("FILTERED")
            return tokens
        pair_xy_value = {"x": -99, "y": -99}
        for key, value in collection.items():
            key_prefix = prefix + "." + str(key)
            value = _load_json(value)
            if is_atomic(value):
                # binning numerical value
                if binning and (key == "x" or key == "y"):
                    n = 2
                    value = int(value/n) * n
                if key in pair_xy_value:
                    pair_xy_value[key] = value
                    if pair_xy:
                        continue
                tokens.append(key_prefix + "." + str(value))
            else:
                tokens.extend(tokenize(value, key_prefix, mode=mode,
                                       filter_player=filter_player, binning=binning, pair_xy=pair_xy))
        if pair_xy and pair_xy_value["x"] >= 0:
            x, y = pair_xy_value["x"], pair_xy_value["y"]
            tokens.append(f"{prefix}.x.{x}.y.{y}")
    else:
        raise NotImplementedError
    return tokens


def generate_dict_prototypes(data: List[Tuple[Dict[Hashable, int], int]], normalized_mix=False) -> List[Dict[Hashable, int]]:
    class_dicts = dict()
    for entry in data:
        if entry[1] not in class_dicts:
            class_dicts[entry[1]] = list()
        b = entry[0] if not normalized_mix else normalize(entry[0])
        class_dicts[entry[1]].append(b)
    prototypes = [Counter() for _ in range(len(class_dicts.keys()))]
    for k, bags in class_dicts.items():
        for b in bags:
            prototypes[k].update(b)
    return prototypes


def filter_tokens(bag: Dict[Hashable, Any], token_filter: List[str]):
    bag_tokens = list(bag.keys())
    for token in bag_tokens:
        for f in token_filter:
            if f.lower() in str(token).lower():
                del bag[token]
                break


class BagToVector:
    def __init__(self, bags: List[Dict[Hashable, float]]):
        self.tokens = set()
        for bag in bags:
            self.tokens.update(bag.keys())
        self.tokens_idx = dict()
        self.tokens_vec = list()
        self.N = len(self.tokens)
        i = 0
        for k in self.tokens:
            self.tokens_idx[k] = i
            self.tokens_vec.append(k)
            i += 1

    def vectorize(self, bag: Dict[Hashable, float]) -> np.ndarray:
        vec = np.zeros(self.N)
        for token in bag.keys():
            if token in self.tokens_idx.keys():
                vec[self.tokens_idx[token]] = bag[token]
        return vec

    def vectorize_with_label(self, bag: Dict[Hashable, float], label: int) -> np.ndarray:
        vec = np.zeros(self.N + 1)
        vec[-1] = label
        for token in bag.keys():
            if token in self.tokens_idx.keys():
                vec[self.tokens_idx[token]] = bag[token]
        return vec

    def get_vec_labels(self):
        return self.tokens_vec
