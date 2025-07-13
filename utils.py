import os
import random
from typing import *
import json
import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from collections import Counter, defaultdict

from distance_matrix_wrapper import DistanceMatrixWrapper
from natsort import os_sorted
from tokenizer import *
import const_dir


# Load json file and return the raw string in oneline (with no pretty print)
def load_from_json(path: str, return_obj=False) -> Union[str, Dict, List]:
    with open(path) as f:
        try:
            data = json.load(f)
        except json.decoder.JSONDecodeError:
            print(path)
            raise json.decoder.JSONDecodeError
    return data if return_obj else json.dumps(data, separators=(',', ':'))


def make_folder(path: str) -> None:
    try:
        os.makedirs(path)
        print(f"Directory '{path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{path}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")


def load_distance_matrix(path: str) -> DistanceMatrixWrapper:
    print("Loading distance matrix from", path, ". . .")
    data = np.load(path)
    return DistanceMatrixWrapper(data)


def load_json_dir(json_dir: str, name="xdd", count=-1, return_obj=False, clip=None):
    dir_scan = os.scandir(json_dir)
    json_entries = list(entry for entry in dir_scan if entry.is_file())
    json_entries = os_sorted(json_entries, key=lambda x: x.name)
    count = min(count, len(json_entries)) if count > 0 else len(json_entries)
    clip = clip if not return_obj else None
    print("Reading {} json_entries from {}, clip={} ...".format(count, name, clip))
    json_objs = list()
    json_names = list()
    for entry in tqdm(json_entries[:count], desc="Reading", disable=False):
        j_obj = load_from_json(entry.path, return_obj=return_obj)
        json_names.append(entry.name)
        if clip is not None:
            json_objs.append(j_obj[:clip])
        else:
            json_objs.append(j_obj)
    return json_objs, json_names


def generate_states_index(states_dirscan: List, path: str, n=-1):
    print("Generating states index and inverse index in {} ...".format(path))
    states_dirscan = os_sorted(states_dirscan, key=lambda x: x.name)
    states_inverse_index_dict = dict()  # Format {gameID:{round:{turn:(filename, index)}}}
    states_index = list()
    i = 0
    n = len(states_dirscan) if n <= 0 else n
    for entry in tqdm(states_dirscan[:n]):
        states_index.append(entry.name)
        s = entry.name.removesuffix(".json").split("-")
        if s[1] not in states_inverse_index_dict:
            states_inverse_index_dict[s[1]] = dict()
        if s[2] not in states_inverse_index_dict[s[1]]:
            states_inverse_index_dict[s[1]][s[2]] = dict()
        states_inverse_index_dict[s[1]][s[2]][s[3]] = (entry.name, i)
        i += 1
    with open(path + "states_inverse_index_dict.json", 'w') as f:
        json.dump(states_inverse_index_dict, f)
    with open(path + "states_index.json", 'w') as f:
        json.dump(states_index, f, indent=2)
    return states_index, states_inverse_index_dict


# Return dict of indexes of states in turn {turn: [index]}
# only consider index up to count or everything if count = -1
def generate_turn_states_dict(states_inverse_index, count=-1, multiple=1):
    turn_states_dict = dict()
    print("Generating turn-state_idx dict...")
    for game_dict in states_inverse_index.values():
        for round_dict in game_dict.values():
            for turn, index in round_dict.items():
                turn = int(turn)
                turn = turn if turn % multiple == 0 else 0
                if turn not in turn_states_dict:
                    turn_states_dict[turn] = list()
                if count == -1 or index[1] < count:
                    turn_states_dict[turn].append(index[1])
    return turn_states_dict


def generate_turn_dist_dict(dist_matrix: DistanceMatrixWrapper, turn_states_dict: Dict, count=-1):
    print("Generating turn-distance dict . . .")
    turn_dist_dict = dict()
    for turn, state_idx in tqdm(turn_states_dict.items()):
        state_idx.sort()
        n = len(state_idx)
        index = 0
        turn_dist_dict[turn] = list()
        for i in range(n):
            if 0 < count <= state_idx[i]:
                break
            for j in range(i + 1, n):
                if 0 < count <= state_idx[j]:
                    break
                # if i == j: continue
                turn_dist_dict[turn].append(dist_matrix.get(state_idx[i], state_idx[j]))
                index += 1
        if len(turn_dist_dict[turn]) == 0:
            del turn_dist_dict[turn]
        else:
            turn_dist_dict[turn] = np.array(turn_dist_dict[turn])
    return turn_dist_dict


def summarize_turn_dist_dict(turn_dist_dict: Dict, plot=True, name="", ax_avg=None, ax_std=None):
    turn_avg_dist_dict = dict()
    turn_std_dist_dict = dict()
    for turn, dist_array in turn_dist_dict.items():
        turn_avg_dist_dict[turn] = float(dist_array.mean())
    for turn, dist_array in turn_dist_dict.items():
        turn_std_dist_dict[turn] = float(dist_array.std())
    if plot:
        turn_avg_dists = list(turn_avg_dist_dict[turn] for turn in sorted(turn_avg_dist_dict.keys()))
        turn_std_dists = list(turn_std_dist_dict[turn] for turn in sorted(turn_std_dist_dict.keys()))

        turn_indices = sorted(turn_avg_dist_dict.keys())

        if ax_avg is None or ax_std is None:
            plt.figure(plt.figure(figsize=(12, 6)))
            plt.subplot(1, 2, 1)
            plt.plot(turn_indices[1:], turn_avg_dists[1:], label=name)
            plt.xlabel("Turn")
            plt.ylabel("Distance")
            plt.title("Average State Distances within a Turn")
            plt.subplot(1, 2, 2)
            plt.plot(turn_indices[1:], turn_std_dists[1:], label=name)
            plt.xlabel("Turn")
            plt.ylabel("Std")
            plt.title("Std of State Distances within a Turn")
        else:
            ax_avg.plot(turn_indices[1:], turn_avg_dists[1:], label=name)
            ax_std.plot(turn_indices[1:], turn_std_dists[1:], label=name)
        # plt.show()
    return turn_avg_dist_dict, turn_std_dist_dict


def load_states_index(path: str) -> Tuple[List[str], dict]:
    try:
        with open(path + "states_inverse_index_dict.json") as f:
            states_inverse_index_dict = json.load(f)
        with open(path + "states_index.json") as f:
            states_index = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("STATES INDEX NOT FOUND!!")
    return states_index, states_inverse_index_dict


def generate_no_and_only_history_states(states_dirscan: List, no_history_path: str, only_history_path: str, n=-1):
    print("Generating game states json with no and only history ...")
    n = len(states_dirscan) if n <= 0 else n
    for entry in tqdm(states_dirscan[:n]):
        with open(entry.path) as f:
            state_dict = json.load(f)
        only_history_dict = dict()
        only_history_dict["historyText"] = state_dict["historyText"]
        del state_dict["historyText"]
        with open(no_history_path + "json/" + entry.name, 'w') as f:
            json.dump(state_dict, f, indent=0)
        with open(only_history_path + "json/" + entry.name, 'w') as f:
            json.dump(only_history_dict, f, indent=2)


def generate_only_history_states(states_dirscan: List, _only_history_path: str, n=-1):
    print("Generating game states json with only history ...")
    n = len(states_dirscan) if n <= 0 else n
    make_folder(_only_history_path)
    for entry in tqdm(states_dirscan[:n]):
        with open(entry.path) as f:
            state_dict = json.load(f)
        only_history_dict = dict()
        only_history_dict["historyText"] = state_dict["historyText"]
        with open(_only_history_path + "json/" + entry.name, 'w') as f:
            json.dump(only_history_dict, f, indent=2)


def generate_no_history_states(states_dirscan: List, _no_history_path: str, n=-1):
    print("Generating game states json with no history ...")
    n = len(states_dirscan) if n <= 0 else n
    _no_history_path += "json/"
    make_folder(_no_history_path)
    for entry in tqdm(states_dirscan[:n]):
        with open(entry.path) as f:
            state_dict = json.load(f)
        del state_dict["historyText"]
        with open(_no_history_path + entry.name, 'w') as f:
            json.dump(state_dict, f, indent=0)


# TODO make this recursive?
# Delete filter_fields from dictionary
def filter_dict_fields(dictionary: Dict, filter_fields: List) -> None:
    for field in filter_fields:
        if field in dictionary:
            del dictionary[field]


# Hide collections that are hidden to everyone
def hide_hidden_info(game_state_dict: Dict, mode="SIZE") -> None:
    for component in game_state_dict.keys():
        value_type = type(game_state_dict[component])
        if value_type is list:
            pass    # TODO implement recursive hide info i.e. look for dicts in the lists
        if value_type is not dict:
            continue
        if "visibility" in game_state_dict[component]:
            if game_state_dict[component]["visibility"] == "HIDDEN_TO_ALL":
                size = len(game_state_dict[component]["components"])
                if mode == "SIZE":  # Convert list of components in collection to its size only
                    game_state_dict[component]["components"] = size
                if mode == "HIDDEN_COMPONENTS":     # Convert each component into a hidden component
                    collection_type = game_state_dict[component]["type"]
                    hidden_str = str(collection_type) + " " + str(component) + " " + "HIDDEN COMPONENT"
                    game_state_dict[component]["components"] = [hidden_str] * size


# Default FOR SeaSaltPaper
# Generate "minimal" game states
def generate_minimal_states(states_dirscan: List, minimal_path: str, filter_fields=None, hide_mode=None, n=-1):
    if filter_fields is None: # Default
        filter_fields = ["lastChance", "protectedHands", "firstPlayer", "playerTotalScores", "gameType", "tick", "roundCounter", "turnCounter",
                         "turnOwner", "nPlayers", "nTeams", "gameStatus", "playerResults"]
    print("Generating minimal game states json, hidden_mode={}  ...".format(hide_mode))
    n = len(states_dirscan) if n <= 0 else n
    for entry in tqdm(states_dirscan[:n]):
        with open(entry.path) as f:
            state_dict = json.load(f)
        filter_dict_fields(state_dict, filter_fields)
        if hide_mode is not None:
            hide_hidden_info(state_dict, mode=hide_mode)
        with open(minimal_path + "json/" + entry.name, 'w') as f:
            json.dump(state_dict, f, indent=2)


# Generate game states hiding info invisible to every player
def generate_hidden_info_states(states_dirscan: List, hidden_info_path: str, hide_mode="SIZE", n=-1):
    print("Generating hidden-info game states json, hidden_mode={}  ...".format(hide_mode))
    n = len(states_dirscan) if n <= 0 else n
    hidden_info_path += "json/"
    make_folder(hidden_info_path)
    for entry in tqdm(states_dirscan[:n]):
        with open(entry.path) as f:
            state_dict = json.load(f)
        hide_hidden_info(state_dict, mode=hide_mode)
        with open(hidden_info_path + entry.name, 'w') as f:
            json.dump(state_dict, f, indent=2)


# Generate game trajectories by concatenating game states json
def generate_game_trajectories(states_inverse_index: Dict, game_name: str, dataset_path: str):
    print("Generating hidden-info game trajectories json  ...")
    game_states_path = dataset_path + "json/"
    trajectories_path = dataset_path + "trajectories/"
    make_folder(trajectories_path)
    for game_id, game_dict in tqdm(states_inverse_index.items()):
        trajectory = list()
        for round, round_dict in sorted(game_dict.items(), key=lambda x: int(x[0])):
            for turn, (filename, i) in sorted(round_dict.items(), key=lambda x: int(x[0])):
                trajectory.append(load_from_json(game_states_path + filename, return_obj=True))
        trajectory_filename = game_name + "-" + str(game_id) + ".json"
        with open(trajectories_path + trajectory_filename, 'w') as f:
            json.dump(trajectory, f)


def compute_compression_ratio(data: List[str], compress: Callable[[ByteString, int], ByteString], level: int):
    compression_ratio = np.zeros(len(data), dtype=np.float64)
    compression_length = np.zeros(len(data), dtype=np.float64)
    for i in tqdm(range(len(data)), desc="Compressing"):
        s = data[i]
        compressed_s = compress(s.encode('utf-8'), level)
        compression_ratio[i] = len(s) / len(compressed_s)
        compression_length[i] = len(compressed_s)
    return compression_ratio, compression_length


def split_data(data: List[Any], test_size: Union[int, float] = 0.25):
    if test_size < 1:
        test_size = int(len(data) * test_size)
    else:
        assert type(test_size) is int
    return data[test_size:], data[:test_size]


def split_x_y(data: List[Any]):
    y = np.array(list(d[-1] for d in data), dtype=int)
    x = list(d[0] for d in data)
    return x, y


def few_shot_split(data: List[Tuple[Any, int]], n_per_class: int):
    class_dicts = dict()
    for d, c in data:
        if c not in class_dicts:
            class_dicts[c] = list()
        class_dicts[c].append((d, c))
    train_data = list()
    test_data = list()
    for c in class_dicts.keys():
        # random.shuffle(class_dicts[c])
        for _ in range(n_per_class):
            train_data.append(class_dicts[c].pop())
        test_data.extend(class_dicts[c])
    random.shuffle(test_data)
    return train_data, test_data


def accuracy(y_true, y_pred, verbose=False):
    if verbose:
        misclassify = dict()
        for pred, true in zip(y_pred, y_true):
            if true == pred:
                continue
            if true in misclassify.keys():
                misclassify[true] += 1
            else:
                misclassify[true] = 1
        print("Misclassification rate:", misclassify)
    return (y_true == y_pred).sum()/len(y_true)


def generate_tokenized_dicts_from_trajectories(trajectories: List[str], trajectories_names: List[str], save_path: str,
                                               ordered=False,
                                               mode: Literal["both", "ordered", "unordered", "char"] = "both"):
    print(f"Generating tokenized dicts from trajectories mode={mode} ...")
    save_path = save_path + const_dir.tokenized_trajectories_paths[mode]
    make_folder(save_path)
    for i in tqdm(range(len(trajectories))):
        if isinstance(trajectories[i], (dict, list)):
            trajectory_obj = trajectories[i]
        else:
            trajectory_obj = json.loads(trajectories[i])
        tokenized_dict = Counter(tokenize(trajectory_obj, ordered=ordered, mode=mode))
        with open(save_path + trajectories_names[i].removesuffix(".json") + '-tokenized.json', 'w') as f:
            json.dump(tokenized_dict, f, indent=4)


def load_class_tokenized_dict(path: str, mode: Literal["both", "unordered", "ordered"] = "both") -> Dict[Hashable, int]:
    try:
        with open(path + f"dataset-tokenized-{mode}.json") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"No dataset-tokenized-{mode}.json found, Creating one ...")
        return generate_class_tokenized_dict(path, mode=mode)


def generate_class_tokenized_dict(path: str, mode: Literal["both", "unordered", "ordered"]) -> Dict[Hashable, int]:
    tokenized_trajectories_path = path + const_dir.tokenized_trajectories_paths[mode]
    token_bags, _ = load_json_dir(tokenized_trajectories_path, path, return_obj=True)
    class_tokenized_dict = Counter()
    for bag in token_bags:
        class_tokenized_dict.update(bag)
    with open(path + f"dataset-tokenized-{mode}.json", 'w') as f:
        json.dump(class_tokenized_dict, f, indent=4)
    return class_tokenized_dict


def get_algo_name(dist, use_prototypes: bool, use_obj: bool):
    name = dist[1] if dist is not None else "NCD"
    name += " Prototype" if (use_prototypes and dist is not None) else " kNN"
    name += " JSON Object" if use_obj else " RAW STRING"
    return name
