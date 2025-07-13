import json
import time

from matplotlib import pyplot as plt
import pandas as pd

from metrics import jensen_shannon_distance, cosine_dist_vec, euclidean_dist_vec, cosine_distance
from vec_main import vec_main
from json_main import json_main
from const import dominion_const, wonders7_const, seasaltpaper_const, cantstop_const, connect4_const, \
    dotsandboxes_const
from typing import *


def get_CI_str(mean, ste):
    range = 1.96 * ste
    return f"{mean:.4f}±{range:.4f}"


if __name__ == '__main__':
    games = [
        # wonders7_const,
        # dominion_const,
        seasaltpaper_const,
        # cantstop_const,
        # connect4_const,
        # dotsandboxes_const
    ]
    n = 1
    dataset_names = ["agent", "param", "seed"]
    dataset_colors = ["red", "blue", "green"]
    seed = None
    use_prototype = True
    normalized_mix_prototype = False
    # tokenize_mode: Literal["both", "ordered", "unordered", "char"] = "unordered"
    num_workers = 8
    game_state_type = "noHistory/"
    weighted = False
    # n_shots_list = list(range(5, 101, 5))
    n_shots_list = [0]  # meaning using the specified splits (i.e. not N-Shot)
    # n_shots_list = [3, 5, 10, 20, 40]
    # n_shots_list = [3, 5]
    dist_vec = [
        # (cosine_dist_vec, "Cosine"),
        (euclidean_dist_vec, "Euclidean")
    ]
    dist_json = [
        # (cosine_distance, "Cosine"),
        # (euclidean_distance, "Euclidian")
        (jensen_shannon_distance, "JSD"),
    ]
    t = time.time()
    verbose = False
    cmap = "Greys"
    if len(n_shots_list) > 1:
        fig_acc, axs_acc = plt.subplots(nrows=1, ncols=len(games), figsize=(6 * len(games), 5), sharey=True)
    result_dict = dict()
    n_shot_result_dict = dict()
    for current_game, game_i in zip(games, range(len(games))):
        datasets = [
            current_game.agent_dataset,
            current_game.param_dataset,
            current_game.seed_dataset if hasattr(current_game, "seed_dataset") else None
        ]
        if datasets[-1] is None:
            datasets = datasets[:-1]
        token_filters = current_game.token_filters if hasattr(current_game, "token_filters") else [None, None, None]
        tokenize_mode = current_game.tokenize_mode
        i = 0
        axs = None
        if len(n_shots_list) <= 1:  # Only plot confusion matrix if not testing n-shot
            n_cols = len(datasets)
            n_rows = 2
            # n_cols = 2
            # n_rows = len(datasets)
            # fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(9.5 * n_cols, n_rows * 7))
            fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(7.5 * n_cols, n_rows * 7))
            fig.suptitle(current_game.actual_name, fontsize=30)
        results_vec = list()
        results_json = list()
        for d, d_name, token_filter in zip(datasets, dataset_names, token_filters):
            dataset_paths, dataset_labels = d
            if len(datasets) == 1:
                ax_vec, ax_json = (axs[0], axs[1]) if axs is not None else (None, None)
            else:
                # ax_vec, ax_json = (axs[i][0], axs[i][1]) if axs is not None else (None, None)
                ax_vec, ax_json = (axs[0][i], axs[1][i]) if axs is not None else (None, None)
            result_vec, ax_vec = vec_main(current_game, dataset_paths, dataset_labels,
                                          dist=dist_vec[0][0], dist_name=dist_vec[0][1], use_prototype=use_prototype,
                                          weighted=weighted, n_shots_list=n_shots_list, n=n,
                                          seed=seed, num_workers=num_workers, verbose=False, ax=ax_vec, cmap=cmap)
            result_json, ax_json = json_main(current_game, dataset_paths, dataset_labels, game_state_type,
                                             dist=dist_json[0][0], dist_name=dist_json[0][1],
                                             use_prototype=use_prototype, use_obj=True,
                                             normalized_mix_prototype=normalized_mix_prototype,
                                             tokenize_mode=tokenize_mode, token_filter_list=token_filter,
                                             weighted=False, n_shots_list=n_shots_list, n=n,
                                             seed=seed, num_workers=num_workers, verbose=verbose, ax=ax_json, cmap=cmap)
            # if len(n_shots_list) <= 1:
            #     result_char, ax_char = json_main(current_game, dataset_paths, dataset_labels, game_state_type,
            #                                      dist=dist_json[0][0], dist_name=dist_json[0][1],
            #                                      use_prototype=use_prototype, use_obj=True,
            #                                      normalized_mix_prototype=normalized_mix_prototype,
            #                                      tokenize_mode="char", n=n, token_filter_list=token_filter,
            #                                      seed=seed, num_workers=num_workers, verbose=verbose, display=False)
            # results_vec.append(result_vec)
            results_json.append(result_json)
            if len(n_shots_list) <= 1:
                label_font = {'size': '22'}
                for ax in (ax_vec, ax_json):
                    ax.tick_params(axis='both', which='major', labelsize=16)
                    if i == 0:
                        ax.set_ylabel("True labels", fontdict=label_font, labelpad=10)
                    else:
                        ax.set_ylabel("")
                ax_json.set_xlabel("Predicted labels", fontdict=label_font, labelpad=10)
                ax_vec.set_xlabel("")
                # for ax in (ax_vec, ax_json):
                #     ax.tick_params(axis='both', which='major', labelsize=15)
                #     if i == len(datasets) - 1:
                #         ax.set_xlabel("Predicted labels", fontdict=label_font, labelpad=10)
                #     else:
                #         ax.set_xlabel("")
                # ax_vec.set_ylabel("True labels", fontdict=label_font, labelpad=10)
                # ax_json.set_ylabel("")
                # TODO also add CI
                # column_name = current_game.game_name + "-" + d_name
                # result_dict[column_name] = list()
                # for r in (result_vec, result_json, result_char):
                #     r = list(r.values())[0]
                #     result_dict[column_name].append(get_CI_str(r[0], r[1]))
                # r = result_json
                # r = list(r.values())[0]
                # result_dict[column_name].append(get_CI_str(r[0], r[1]))
            i += 1
        if len(n_shots_list) <= 1:
            mid_i = 1
            axs[0][mid_i].set_title("Hand-crafted", fontsize=25, pad=15)
            axs[1][mid_i].set_title("JSON-Bag", fontsize=25, pad=15)
            # axs[0].set_title("Hand-crafted", fontsize=22, pad=15)
            # axs[1].set_title("JSON-Bag", fontsize=22, pad=15)
            # if len(datasets) == 1:
            #     axs[0].set_title("Hand-crafted", fontsize=22, pad=15)
            #     axs[1].set_title("JSON-Bag", fontsize=22, pad=15)
            # else:
            #     axs[0][0].set_title("Hand-crafted", fontsize=22, pad=15)
            #     axs[0][1].set_title("JSON-Bag", fontsize=22, pad=15)
            fig.savefig(f"{current_game.game_name}-ConfusionMatrices.png", dpi=200)
            # fig.savefig(f"{current_game.game_name}-agent-ConfusionMatrices.png")
        # else:
        #     fig, ax = plt.subplots(figsize=(7, 5))
        #     # for r_vec, r_json, name, color in zip(results_vec, results_json, dataset_names, dataset_colors):
        #     #     n_shot_result_dict[current_game.game_name + "-" + name] = [r_vec, r_json]
        #     #     ax = axs_acc.flatten()[game_i]
        #     #     accuracy_vec = list(r_vec[n_shots][0] for n_shots in n_shots_list)
        #     #     ax.plot(n_shots_list, accuracy_vec, label=name + "-handcrafted", ls='--', color=color)
        #     #     accuracy_json = list(r_json[n_shots][0] for n_shots in n_shots_list)
        #     #     ax.plot(n_shots_list, accuracy_json, label=name + "-json", color=color)
        #     #     ax.set_title(current_game.actual_name, fontsize=18, pad=15)
        #     #     ax.tick_params(axis='both', which='major', labelsize=12)
        #     #     ax.set_xlabel("Training samples per label", fontsize=13, labelpad=5)
        #     #     if game_i == 0:
        #     #         ax.set_ylabel("Accuracy", fontsize=15, labelpad=10)
        #     #     if game_i == 1:
        #     #         ax.legend(prop={'size': 14})
        #     for r_json, name, color in zip(results_json, dataset_names, dataset_colors):
        #         n_shot_accuracy = list()
        #         for n_shot in n_shots_list:
        #             n_shot_accuracy.append(get_CI_str(r_json[n_shot][0], r_json[n_shot][1]))
        #         n_shot_result_dict[current_game.game_name + "-" + name] = n_shot_accuracy

    # plt.legend()
    if len(n_shots_list) > 1:
        # with open("n-shots.json", "w") as f:
        #     json.dump(n_shot_result_dict, f, indent=4)
        # fig_acc.savefig("NShotAccuracy.png")
        # fig_acc.show()
        df_nshot = pd.DataFrame.from_dict(n_shot_result_dict)
        df_nshot = df_nshot.rename(index={i: f"{n_shot}-Shot" for i, n_shot in zip(range(len(n_shots_list)), n_shots_list)})
        # df_nshot.to_csv("n_shot_table.csv")
    else:
        result_pd = pd.DataFrame.from_dict(result_dict)
        # result_pd = result_pd.rename(index={0: "Hand-crafted", 1: "JSON-Bag", 2: "JSON-Char"})
        # result_pd = result_pd.rename(index={0: "JSON-Cosine"})
        # result_pd.to_csv("Results-JSON-Cosine.csv")
        plt.show()
    print("FINISHED IN " + str(time.time() - t) + " SECONDS")
