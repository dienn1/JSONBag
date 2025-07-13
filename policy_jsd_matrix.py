import os
import random
from typing import *

import matplotlib
import numpy as np

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_selection import r_regression

from const import seasaltpaper_const, wonders7_const, dominion_const, cantstop_const, connect4_const, dotsandboxes_const
from jsd_analysis import jsd_class_matrix


def rename_columns(df):
    column_names = df.columns.tolist()
    new_column_names = []
    for i in range(len(column_names)):
        if "64" in column_names[i]:
            new_column_names.append("mcts64")
        elif "128" in column_names[i]:
            new_column_names.append("mcts128")
        elif column_names[i] == "mcts1":
            new_column_names.append("mcts64v")
        elif "random" in column_names[i]:
            new_column_names.append("random")
        elif "osla" in column_names[i]:
            new_column_names.append("osla")
        else:
            raise AssertionError("INVALID COLUMN NAME {}".format(column_names[i]))
    return df.rename(columns=dict(zip(column_names, new_column_names)))


def process_policy_jsd_matrix(csv_path):
    order = ["mcts64v", "mcts64", "mcts128", "osla", "random"]
    df = pd.read_csv(csv_path)
    df = rename_columns(df)
    df = df.rename(index={i: df.columns[i] for i in range(len(df.columns))})
    normalize_const = df["random"]["random"]
    for row in order:
        for col in order:
            df.at[row, col] = df.loc[row, col] / normalize_const
    return df.reindex(columns=order, index=order)


if __name__ == '__main__':
    # pm = process_policy_jsd_matrix("AgreementMatrix.csv")
    # print(pm)
    # sns.heatmap(pm, annot=True, fmt=".2f", cmap="viridis")
    games = {
        wonders7_const: "unordered",
        dominion_const: "unordered",
        seasaltpaper_const: "unordered",
        cantstop_const: "both",
        connect4_const: "ordered",
        dotsandboxes_const: "ordered"
    }
    pearson_r_dict = dict()
    markers = matplotlib.lines.Line2D.markers.keys()
    markers = list(markers)
    markers = markers[1:]
    csv_name = "PolicyJSDMatrix.csv"
    overall_jsd_list = list()
    overall_policy_jsd_list = list()
    for current_game, tokenize_mode, m in zip(games.keys(), games.values(), markers):
        policy_matrix_path = f"results/{current_game.game_name}/{csv_name}"
        pm = process_policy_jsd_matrix(policy_matrix_path)
        agent_dataset = current_game.agent_dataset
        dataset_paths, dataset_labels = agent_dataset
        game_state_type = "noHistory/"
        # tokenize_mode: Literal["both", "ordered", "unordered", "char"] = "both"
        token_filters = current_game.token_filters if hasattr(current_game, "token_filters") else [None, None, None]
        jsd_matrix = jsd_class_matrix(dataset_paths, dataset_labels, game_state_type, tokenize_mode, token_filters[0])
        np_policy_matrix = pm.to_numpy()
        print(np_policy_matrix)
        print(pm)
        print(jsd_matrix)
        policy_jsd_list = list()
        jsd_list = list()
        n = jsd_matrix.shape[0]
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    continue
                policy_jsd_list.append(np_policy_matrix[i][j])
                jsd_list.append(jsd_matrix[i][j])
        overall_policy_jsd_list.extend(policy_jsd_list)
        overall_jsd_list.extend(jsd_list)
        policy_jsd_list = np.array(policy_jsd_list)
        jsd_list = np.array(jsd_list)
        plt.scatter(jsd_list, policy_jsd_list, marker=m, s=50, label=current_game.actual_name)
        pearson_r = r_regression(jsd_list.reshape(-1, 1), policy_jsd_list)
        pearson_r_dict[current_game.game_name] = round(pearson_r[0], 4)
    overall_policy_jsd_list = np.array(overall_policy_jsd_list)
    overall_jsd_list = np.array(overall_jsd_list)
    print(len(overall_policy_jsd_list))
    pearson_r = r_regression(overall_jsd_list.reshape(-1, 1), overall_policy_jsd_list)
    pearson_r_dict["overall"] = round(pearson_r[0], 4)
    print("----------------------------------\n\nJSON-Policy JSD Pearson Correlation:")
    for game_name, r in pearson_r_dict.items():
        print(game_name, r)
    pearson_df = pd.DataFrame.from_dict(pearson_r_dict, orient="index")
    pearson_df = pearson_df.rename(columns={0: "Pearson-R"})
    p = plt.figure(figsize=(10, 10))
    plt.xlabel("JSON-Bag Prototype JSD")
    plt.ylabel("Policy Distance")
    plt.legend()

    # pearson_df.to_csv("results/JSON-Policy-pearsonR.csv")
    # plt.savefig("results/JSON-Policy plot.png", dpi=200)
    plt.show()
