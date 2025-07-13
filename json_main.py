import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics as sk_metrics
from typing import *
from collections import Counter

import const_dir
from knn import knn_test, nn_prototype_test
from const import wonders7_const, cantstop_const, dotsandboxes_const, connect4_const, dominion_const, seasaltpaper_const
from tokenizer import filter_tokens, tokenize
from utils import load_json_dir
from metrics import jensen_shannon_distance, analyze_jensen_shannon_distance


def json_main(game_const, data_paths, data_labels, game_state_type,
              dist: Callable[[Any, Any], float] = jensen_shannon_distance, dist_name="",
              use_prototype=True, use_obj=True, normalized_mix_prototype=False,
              tokenize_mode: Literal["both", "ordered", "unordered", "char"] = "unordered",
              token_filter_list: List[str] = None,
              clip=None, count=-1,
              weighted=False, n_shots_list=None, n=1,
              seed=None, num_workers=0, verbose=False, display=True, ax=None, create_colorbar=False, cmap="viridis"):
    data = list()
    print("READING DATASET FROM {}".format(data_paths))
    per_class_data = list()

    for i in range(len(data_paths[:])):
        path = data_paths[i] + game_state_type
        json_dir = (path + const_dir.tokenized_trajectories_paths[tokenize_mode]) if use_obj else path + "trajectories/"
        if tokenize_mode == "char" and token_filter_list is not None:
            json_dir = path + "trajectories/"
        print(json_dir)
        trajectories, trajectories_name = load_json_dir(json_dir, path, count=count, return_obj=use_obj,
                                                        clip=clip)
        if use_obj and token_filter_list is not None:
            if tokenize_mode == "char":     # MANUALLY FILTER SHIT FOR CHAR TOKENIZE
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

    # analyze_jensen_shannon_distance(per_class_data[0], per_class_data[2], top_k=10)
    # return None, None

    assert len(data) > 0
    _tokenize = False if dist is None or "compress" in dist_name else True
    print(f"Using {dist_name} ...")
    conf_matrix_agg = None
    n_shots_list = n_shots_list if n_shots_list is not None else [0]
    n_shots_results = dict()  # {n_shots: (accuracy, ste)}
    test_size = 0.5
    for n_shots in n_shots_list:
        if n_shots > 0:
            print("CLASSIFYING {}-SHOT".format(n_shots))
        else:
            print(f"CLASSIFYING WITH {int(len(data) * (1 - test_size))} training samples")
        valid_accuracy = list()
        test_accuracy = list()
        for _ in range(n):
            if not use_prototype:
                valid, test, conf_matrix = knn_test(data, dist=dist, seed=seed, num_workers=num_workers,
                                                    weighted=weighted,
                                                    train_split=0.25, test_split=0.5, n_shots=n_shots,
                                                    verbose=verbose)
                valid_accuracy.append(valid)
            else:
                test, conf_matrix, decision_margins = nn_prototype_test(data, dist=dist,
                                                                        normalized_mix=normalized_mix_prototype,
                                                                        seed=seed,
                                                                        test_size=test_size, n_shots=n_shots,
                                                                        verbose=verbose)
            conf_matrix_agg = conf_matrix if conf_matrix_agg is None else conf_matrix_agg + conf_matrix
            test_accuracy.append(test)

        valid_accuracy = np.array(valid_accuracy)
        test_accuracy = np.array(test_accuracy)
        mean = test_accuracy.mean()
        ste = test_accuracy.std() / np.sqrt(len(test_accuracy))
        if verbose:
            if not use_prototype:
                print("Validation Accuracy Mean:", valid_accuracy.mean())
                print("Validation Accuracy Max:", valid_accuracy.max())
                print("Validation Accuracy Min:", valid_accuracy.min())
                print("Validation Accuracy Std:", valid_accuracy.std())
        print("Test Accuracy Mean:", test_accuracy.mean())
        print("Test Accuracy Max:", test_accuracy.max())
        print("Test Accuracy Min:", test_accuracy.min())
        print("Test Accuracy std:", test_accuracy.std())
        print("Test Accuracy 95% CI:", (mean - 1.96 * ste, mean + 1.96 * ste))
        print("-------------")
        n_shots_results[n_shots] = (mean, ste)
    if len(n_shots_list) == 1 and display:
        n_shots = n_shots_list[0]
        if ax is None:
            fig, ax = plt.subplots(figsize=(8.5, 6))
            algo_name = (("Prototype" if use_prototype else "kNN") + " " + dist_name + "\n"
                         + (f"{n_shots}-shot" if n_shots > 0 else "") + (" RawJSON" if not use_obj else "BagJSON"))
            fig.suptitle(game_const.game_name + f" json {tokenize_mode} {game_state_type.strip('/')} " + algo_name,
                         fontsize=16)
        disp = sk_metrics.ConfusionMatrixDisplay(conf_matrix_agg / n, display_labels=data_labels).plot(ax=ax,
                                                                                                       colorbar=create_colorbar,
                                                                                                       values_format=".2f",
                                                                                                       cmap=cmap)
        disp.im_.set_clim(0, 1)
        for values in disp.text_.ravel():
            values.set_fontsize(20)
        return n_shots_results, ax
    return n_shots_results, None


if __name__ == '__main__':
    # current_game = wonders7_const
    current_game = dominion_const
    # current_game = seasaltpaper_const
    # current_game = cantstop_const
    # current_game = connect4_const
    # current_game = dotsandboxes_const
    datasets = {
        "agent": current_game.agent_dataset,
        "param": current_game.param_dataset,
        "seed": current_game.seed_dataset
    }
    tokenize_mode: Literal["both", "ordered", "unordered", "char"] = "unordered"
    seed = None
    normalized_mix_prototype = False
    # clip = 120_000
    clip = None
    count = -1
    num_workers = 8
    # game_state_type = "noHistoryHiddenInfo/"
    game_state_type = "noHistory/"
    dist = [
        # (None, "NCD"),
        # (cosine_distance, "Cosine"),
        # (euclidean_distance, "Euclidian")
        (jensen_shannon_distance, "JSD"),
        # (total_variation_distance, "VarDist")
        # (CompressCE(compress_algos[0][0], levels[0]).compute, "CompressCE " + compress_name[0])
    ]
    axs = list()
    results = list()
    # n_shots_list = list(range(5, 101, 5))
    n_shots_list = [0]
    n = 10
    verbose = False
    token_filters = current_game.token_filters if hasattr(current_game, "token_filters") else [None, None, None]
    print(token_filters)
    use_prototype = True
    use_obj = True
    n_cols = len(datasets)
    n_rows = 1
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(8 * n_cols, n_rows * 8))
    fig.suptitle(current_game.game_name + f" {dist[0][1]} Prototype Confusion Matrix", fontsize=30)
    label_font = {'size': '20'}
    t = time.time()
    i = 0
    for d, token_filter in zip(datasets.values(), token_filters):
        dataset_paths, dataset_labels = d
        ax = axs.flatten()[i]
        result, _ = json_main(current_game, dataset_paths, dataset_labels, game_state_type,
                               dist=dist[0][0], dist_name=dist[0][1],
                               use_prototype=use_prototype, use_obj=use_obj,
                               normalized_mix_prototype=normalized_mix_prototype,
                               tokenize_mode=tokenize_mode, token_filter_list=token_filter,
                               clip=clip, count=count,
                               weighted=False, n_shots_list=n_shots_list, n=n,
                               seed=seed, num_workers=num_workers, verbose=verbose, ax=ax)
        ax.tick_params(axis='both', which='major', labelsize=16)
        if i == 0:
            ax.set_ylabel("True labels", fontdict=label_font, labelpad=20)
        else:
            ax.set_ylabel("")
        ax.set_xlabel("Predicted labels", fontdict=label_font, labelpad=15)
        print("---------------------------------------------------*---------------------------------------------------")
        i += 1
        # axs.append(ax)
        results.append(result)
    if len(n_shots_list) > 1:
        for r, name in zip(results, datasets.keys()):
            accuracy = list(r[n_shots][0] for n_shots in n_shots_list)
            plt.plot(n_shots_list, accuracy, label=name)
            for r, name in zip(results, datasets.keys()):
                accuracy = list(r[n_shots][0] for n_shots in n_shots_list)
                plt.plot(n_shots_list, accuracy, label=name)
        plt.legend()
        plt.xlabel("TRAINING SAMPLES PER LABEL")
        plt.ylabel("Accuracy")
        plt.title(f"{current_game.game_name} JSON {dist[0][1]} \nN-SHOT CLASSIFICATION")
    plt.show()
    print("FINISHED IN " + str(time.time() - t) + " SECONDS")
