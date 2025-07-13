import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from collections import Counter

from typing import *
import const_dir
from const import wonders7_const, dominion_const, seasaltpaper_const, connect4_const, cantstop_const, dotsandboxes_const
from metrics import analyze_jensen_shannon_distance
from tokenizer import filter_tokens
from utils import load_json_dir


def jsd_class_matrix(data_paths, data_labels, game_state_type,
                     tokenize_mode: Literal["both", "ordered", "unordered", "char"] = "unordered",
                     token_filter_list=None):
    data = list()
    per_class_data = list()
    print("READING DATASET FROM {}".format(data_paths))
    for i in range(len(data_paths[:])):
        path = data_paths[i] + game_state_type
        json_dir = (path + const_dir.tokenized_trajectories_paths[tokenize_mode])
        print(json_dir)
        trajectories, trajectories_name = load_json_dir(json_dir, path, return_obj=True)
        if token_filter_list is not None:
            for bag in trajectories:
                filter_tokens(bag, token_filter_list)
        data.extend([(t, i) for t in trajectories])
        per_class_data.append(trajectories)
    num_class = len(data_labels)
    jsd_matrix = np.zeros((num_class, num_class))
    for i in range(num_class):
        for j in range(i, num_class):
            jsd_matrix[i, j], _ = analyze_jensen_shannon_distance(per_class_data[i], per_class_data[j], verbose=False)
    jsd_matrix += jsd_matrix.T
    return jsd_matrix


def jsd_class_prototype_analysis(data_paths, class_names, game_state_type,
                                 tokenize_mode: Literal["both", "ordered", "unordered", "char"],
                                 token_filter_list=None, top_k=10):
    data = list()
    print("READING DATASET FROM {}".format(data_paths))
    per_class_data = list()

    for i in range(len(data_paths[:])):
        path = data_paths[i] + game_state_type
        json_dir = path + const_dir.tokenized_trajectories_paths[tokenize_mode]
        if tokenize_mode == "char" and token_filter_list is not None:
            json_dir = path + "trajectories/"
        print(json_dir)
        trajectories, trajectories_name = load_json_dir(json_dir, path, count=-1, return_obj=True)
        if token_filter_list is not None:
            if tokenize_mode == "char":  # MANUALLY FILTER SHIT FOR CHAR TOKENIZE
                tokenized_trajectories = list()
                for trajectory in trajectories:
                    for s in trajectory:
                        key_list = list(s.keys())
                        for key in key_list:
                            for token in token_filter_list:
                                if token.lower() in key.lower():
                                    del s[key]
                                    # print("delete", key)
                    tokenized_trajectories.append(Counter(str(trajectory)))
                trajectories = tokenized_trajectories
            else:
                for bag in trajectories:
                    filter_tokens(bag, token_filter_list)
        data.extend([(t, i) for t in trajectories])
        per_class_data.append(trajectories)

    for i in range(len(per_class_data)):
        for j in range(i, len(per_class_data)):
            if i == j:
                continue
            print(f"JSD ANALYSIS BETWEEN {class_names[i]} and {class_names[j]}")
            analyze_jensen_shannon_distance(per_class_data[i], per_class_data[j], top_k=top_k)
            print("------------------------------------------")


if __name__ == "__main__":
    # current_game = wonders7_const
    # current_game = dominion_const
    # current_game = seasaltpaper_const
    # current_game = cantstop_const
    # current_game = connect4_const
    current_game = dotsandboxes_const
    datasets = {
        "agent": current_game.agent_dataset,
        # "param": current_game.param_dataset,
        # "seed": current_game.seed_dataset
    }
    game_state_type = "noHistory/"
    # tokenize_mode: Literal["both", "ordered", "unordered", "char"] = "unordered"
    tokenize_mode = current_game.tokenize_mode
    token_filters = current_game.token_filters if hasattr(current_game, "token_filters") else [None, None, None]
    n_cols = len(datasets)
    n_rows = 1
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(6.5 * n_cols, n_rows * 6))
    fig.suptitle(current_game.actual_name + " JSD Matrix", fontsize=20)
    i = 0
    for d, token_filter in zip(datasets.values(), token_filters):
        dataset_paths, dataset_labels = d
        jsd_matrix = jsd_class_matrix(dataset_paths, dataset_labels, game_state_type, tokenize_mode, token_filter)
        ax = axs.flatten()[i] if n_cols > 1 else axs
        sns.heatmap(jsd_matrix, annot=True, fmt=".2f", cmap="Greys_r",
                    xticklabels=dataset_labels, yticklabels=dataset_labels, ax=ax,
                    vmin=0, vmax=0.15, cbar=False,
                    annot_kws={"size": 16})
        ax.tick_params(axis='both', which='major', labelsize=15)
        i += 1
        jsd_class_prototype_analysis(dataset_paths, dataset_labels, game_state_type, tokenize_mode, token_filter, top_k=10)
    # plt.savefig("DotsAndBoxes-JSDMatrix.png", dpi=200)
    plt.show()
