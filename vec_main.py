import time
from typing import *

import numpy as np
from matplotlib import pyplot as plt
from metrics import cosine_dist_vec, euclidean_dist_vec
from sklearn import metrics as sk_metrics

from const import seasaltpaper_const, connect4_const, dotsandboxes_const, wonders7_const, dominion_const, cantstop_const
from data_vec_utils import process_data
from knn import knn_test_vec, nn_prototype_vec_test


def vec_main(game_const, data_paths, data_labels,
             dist: Callable[[np.ndarray, np.ndarray], float], dist_name, use_prototype=True,
             weighted=False, n_shots_list=None, n=1,
             seed=None, num_workers=0, verbose=False, ax=None, create_colorbar=False, cmap="viridis"):
    data = list()
    print("READING DATASET FROM {}".format(data_paths))
    for i in range(len(data_paths)):
        data.extend(list(process_data(data_paths[i] + "GAME_OVER.csv", game_const.features, game_const.n_players, label=i)))
    data = np.array(data)
    print(data.shape)
    print(data[0])
    conf_matrix_agg = None

    n_shots_list = n_shots_list if n_shots_list is not None else [0]
    n_shots_results = dict()  # {n_shots: (accuracy, ste)}
    test_size = 0.5
    for n_shots in n_shots_list:
        if n_shots > 0:
            print("CLASSIFYING {}-SHOT".format(n_shots))
        else:
            print(f"CLASSIFYING WITH {int(len(data) * (1-test_size))} training samples")
        test_accuracy = list()
        valid_accuracy = list()
        for i in range(n):
            if use_prototype:
                test, conf_matrix, decision_margins = nn_prototype_vec_test(data, dist=dist, seed=seed,
                                                                            test_size=test_size, n_shots=n_shots,
                                                                            verbose=verbose)
            else:
                valid, test, conf_matrix = knn_test_vec(data, dist=dist, seed=seed, num_workers=num_workers,
                                                        weighted=weighted,
                                                        train_split=0.25, test_split=0.5, n_shots=n_shots,
                                                        verbose=verbose)
                valid_accuracy.append(valid)
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
    # Plot Confusion Matrix
    if len(n_shots_list) == 1:
        n_shots = n_shots_list[0]
        if ax is None:
            fig, ax = plt.subplots(figsize=(8.5, 6))
            algo_name = ("Prototype" if use_prototype else "kNN") + " " + dist_name + (f"\n{n_shots}-shot" if n_shots > 0 else "")
            fig.suptitle(game_const.game_name + " handcraft " + algo_name, fontsize=16)
        disp = sk_metrics.ConfusionMatrixDisplay(conf_matrix_agg / n, display_labels=data_labels).plot(ax=ax, colorbar=create_colorbar,
                                                                                              values_format=".2f", cmap=cmap)
        disp.im_.set_clim(0, 1)
        for values in disp.text_.ravel():
            values.set_fontsize(20)
        return n_shots_results, ax
    return n_shots_results, None


if __name__ == '__main__':
    # current_game = wonders7_const
    # current_game = dominion_const
    # current_game = seasaltpaper_const
    # current_game = connect4_const
    # current_game = dotsandboxes_const
    current_game = cantstop_const
    datasets = [
        current_game.agent_dataset,
        current_game.param_dataset,
        current_game.seed_dataset
    ]
    dataset_names = ["agent", "param", "seed"]
    axs = list()
    results = list()
    seed = None
    num_workers = 0
    weighted = False
    # n_shots_list = list(range(5, 101, 5))
    n_shots_list = [0]
    dist = [
        # (cosine_dist_vec, "Cosine"),
        (euclidean_dist_vec, "Euclidean")
    ]
    use_prototype = True
    t = time.time()
    n = 10
    for d in datasets[:]:
        dataset_paths, dataset_labels = d
        result, ax = vec_main(current_game, dataset_paths, dataset_labels,
                              dist=dist[0][0], dist_name=dist[0][1], use_prototype=use_prototype,
                              weighted=weighted, n_shots_list=n_shots_list, n=n,
                              seed=seed, num_workers=num_workers, verbose=False)
        print("---------------------------------------------------*---------------------------------------------------")
        axs.append(ax)
        results.append(result)
    if len(n_shots_list) > 1:
        for r, name in zip(results, dataset_names):
            accuracy = list(r[n_shots][0] for n_shots in n_shots_list)
            plt.plot(n_shots_list, accuracy, label=name)
        plt.legend()
        plt.xlabel("TRAINING SAMPLES PER LABEL")
        plt.ylabel("Accuracy")
        plt.title(f"{current_game.game_name} Handcraft {dist[0][1]} \nN-SHOT CLASSIFICATION")
    plt.show()
    print("FINISHED IN " + str(time.time() - t) + " SECONDS")
