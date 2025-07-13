import os
import random
import time
from copy import copy
from typing import *
from collections import Counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model._cd_fast import ConvergenceWarning
from sklearn.model_selection import train_test_split
import sklearn.metrics as sk_metrics
import warnings
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import const_dir
from knn import NearestPrototypeSearch
from analysis_main import get_CI_str
from const import wonders7_const, cantstop_const, seasaltpaper_const, connect4_const, \
    dotsandboxes_const, dominion_const
from data_vec_utils import process_data, FeaturesRangeNormalizer
from metrics import cosine_dist_vec, euclidean_dist_vec
from tokenizer import filter_tokens
from utils import accuracy, load_json_dir, split_data, split_x_y


def any_model_test(data, data_labels, game_name,
                   model, model_name, scale_mode=None,
                   feature_names=None, display=True,
                   n=1, ax=None, cmap="viridis"):
    conf_matrix_agg = None
    test_accuracy = list()
    data_type = "VECTORIZED JSON" if isinstance(data[0], tuple) else "HANDCRAFTED"
    dict_data = isinstance(data[0], tuple)
    feature_importances = Counter() if dict_data else np.zeros(len(feature_names))
    for i in range(n):
        if dict_data:
            random.shuffle(data)
            train_data, test_data = split_data(data, test_size=0.5)
            x_train, y_train = split_x_y(train_data)
            x_test, y_test = split_x_y(test_data)
            vectorizer = DictVectorizer()
            x_train = vectorizer.fit_transform(x_train)
            x_test = vectorizer.transform(x_test)
            x_train = x_train.toarray()
            x_test = x_test.toarray()
            # print(x_train.shape, x_test.shape)
            feature_names = vectorizer.feature_names_
        else:
            X, y = data[:, :-1], data[:, -1]
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=None)
        if scale_mode is not None:
            if scale_mode == "range":
                scaler = FeaturesRangeNormalizer(x_train)
                x_train = scaler.normalize(x_train)
                x_test = scaler.normalize(x_test)
            if scale_mode == "standard":
                scaler = StandardScaler()
                scaler.fit(x_train)
                x_train = scaler.transform(x_train)
                x_test = scaler.transform(x_test)
            if scale_mode == "row_normalized":
                x_train = x_train / x_train.sum(axis=1, keepdims=True)
                # print(x_train.sum())
                x_test = x_test / x_test.sum(axis=1, keepdims=True)
                # print(x_test.sum())
        # print(x_test[0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x_train, y_train)
        if hasattr(model, "feature_importances_"):
            if dict_data:
                feature_importances_dict = {name: importance for name, importance in
                                            zip(feature_names, model.feature_importances_)}
                feature_importances.update(feature_importances_dict)
            else:
                feature_importances += model.feature_importances_
        y_pred = model.predict(x_test)
        # print(y_pred)
        test_accuracy.append(accuracy(y_test, y_pred))
        conf_matrix = sk_metrics.confusion_matrix(y_test, y_pred, normalize="true")
        conf_matrix_agg = conf_matrix if conf_matrix_agg is None else conf_matrix_agg + conf_matrix
    test_accuracy = np.array(test_accuracy)
    mean = test_accuracy.mean()
    ste = test_accuracy.std() / np.sqrt(len(test_accuracy))
    print("Test Accuracy Mean:", test_accuracy.mean())
    print("Test Accuracy Max:", test_accuracy.max())
    print("Test Accuracy Min:", test_accuracy.min())
    print("Test Accuracy std:", test_accuracy.std())
    print("Test Accuracy 95% CI:", (mean - 1.96 * ste, mean + 1.96 * ste))
    if not dict_data:
        feature_importances = {name: importance for name, importance in zip(feature_names, feature_importances)}
    top_k = min(10, len(feature_importances))
    sorted_features = sorted(feature_importances, key=feature_importances.get, reverse=True)
    print("FEATURE IMPORTANCES")
    for i in range(top_k):
        print(sorted_features[i], feature_importances[sorted_features[i]])
    print("-------------")

    if display:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8.5, 6))
            algo_name = model_name
            fig.suptitle(game_name + f" {data_type} " + algo_name, fontsize=16)
        disp = sk_metrics.ConfusionMatrixDisplay(conf_matrix_agg / n, display_labels=data_labels).plot(ax=ax,
                                                                                                       colorbar=False,
                                                                                                       values_format=".2f",
                                                                                                       cmap=cmap)
        disp.im_.set_clim(0, 1)
        for values in disp.text_.ravel():
            values.set_fontsize(16)
        return test_accuracy, ax
    return test_accuracy, None


def load_csv_data(game_const, data_paths):
    data = list()
    print("READING DATASET FROM {}".format(data_paths))
    for i in range(len(data_paths)):
        data.extend(
            list(process_data(data_paths[i] + "GAME_OVER.csv", game_const.features, game_const.n_players, label=i)))
    data = np.array(data)
    print(data.shape)
    return data


def load_json_bag(data_paths, tokenize_mode, game_state_type, token_filter_list=None):
    data = list()
    print("READING DATASET FROM {}".format(data_paths))
    per_class_data = list()
    for i in range(len(data_paths[:])):
        path = data_paths[i] + game_state_type
        json_dir = (path + const_dir.tokenized_trajectories_paths[tokenize_mode])
        json_dir2 = (path + "tokenizedTrajectories/")
        os.rename(json_dir2, json_dir)
        if tokenize_mode == "char" and token_filter_list is not None:
            json_dir = path + "trajectories/"
        print(json_dir)
        trajectories, trajectories_name = load_json_dir(json_dir, path, return_obj=True)
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
    return data


if __name__ == "__main__":
    games = [
        wonders7_const,
        # dominion_const,
        # seasaltpaper_const,
        # cantstop_const,
        # connect4_const,
        # dotsandboxes_const
    ]
    n = 1
    model = [
        (RandomForestClassifier(n_estimators=100), "RandomForestClassifier"),
        # (GradientBoostingClassifier(), "GradientBoostingClassifier"),
        # (xgb.XGBClassifier(n_estimators=100), "XGBClassifier")
        # (LogisticRegression(max_iter=200, penalty="elasticnet", solver="saga", l1_ratio=0.7), "LogisticRegression"),
        # (LogisticRegression(max_iter=200), "LogisticRegression"),
        # (LinearDiscriminantAnalysis(), "LinearDiscriminantAnalysis"),
        # (NearestPrototypeSearch(cosine_dist_vec), "CosineDistance"),
        # (NearestPrototypeSearch(euclidean_dist_vec), "EuclideanDistance"),
        # (NearestPrototypeSearch(jensenshannon), "JSD")
    ]
    dataset_names = ["agent", "param", "seed"]
    # tokenize_mode: Literal["both", "ordered", "unordered", "char"] = "char"
    game_state_type = "noHistory/"
    # scale_mode = "row_normalized"
    # scale_mode = "range"
    # scale_mode = "standard"
    # scale_mode = None
    label_font = {'size': '20'}
    cmap = "Greys"
    result_dict = dict()
    t = time.time()
    for current_game in games:
        datasets = [
            current_game.agent_dataset,
            current_game.param_dataset,
            current_game.seed_dataset if hasattr(current_game, "seed_dataset") else None
        ]
        if datasets[-1] is None:
            datasets = datasets[:-1]
        token_filters = current_game.token_filters if hasattr(current_game, "token_filters") else [None, None, None]
        feature_names = copy(current_game.features) if hasattr(current_game, "features") else None
        if (feature_names is not None) and (not hasattr(current_game, "no_score")):
            num_players = current_game.n_players
            feature_names.extend(f"player{i}_score" for i in range(num_players))
            for i in range(num_players):
                feature_names.append(f"player{i}_score_intercept")
                feature_names.append(f"player{i}_score_slope")
        # n_cols = len(datasets)
        # n_rows = 1
        # fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(8 * n_cols, n_rows * 8))
        # fig.suptitle(current_game.actual_name + f" {model[0][1]} Confusion Matrix", fontsize=30)
        i = 0
        # n_cols = 2
        # n_rows = len(datasets)
        n_cols = len(datasets)
        n_rows = 2
        # fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(8.5 * n_cols, n_rows * 8))
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(9.5 * n_cols, n_rows * 7))
        fig.suptitle(current_game.actual_name + " " + model[0][1], fontsize=22)
        # axs = None
        for d, d_name, token_filter in zip(datasets, dataset_names, token_filters):
            dataset_paths, dataset_labels = d
            if len(datasets) == 1:
                ax_vec, ax_json = (axs[0], axs[1]) if axs is not None else (None, None)
            else:
                # ax_vec, ax_json = (axs[i][0], axs[i][1]) if axs is not None else (None, None)
                ax_vec, ax_json = (axs[0][i], axs[1][i]) if axs is not None else (None, None)
            if feature_names is None:
                result_vec = None
            else:
                data = load_csv_data(current_game, dataset_paths)
                result_vec, ax_vec = any_model_test(data, dataset_labels, current_game.game_name,
                                                    scale_mode=None,
                                                    feature_names=feature_names,
                                                    model=model[0][0], model_name=model[0][1], n=n, ax=ax_vec,
                                                    cmap=cmap,
                                                    display=False if ax_vec is None else True)
            tokenize_mode = current_game.tokenize_mode
            # ax_json = axs
            data = load_json_bag(dataset_paths, tokenize_mode=tokenize_mode, game_state_type=game_state_type,
                                 token_filter_list=token_filter)
            result_json, ax_json = any_model_test(data, dataset_labels, current_game.game_name,
                                                  scale_mode="range",
                                                  feature_names=feature_names,
                                                  model=model[0][0], model_name=model[0][1], n=n, ax=ax_json,
                                                  cmap=cmap,
                                                  display=False if ax_json is None else True)

            # data = load_json_bag(dataset_paths, tokenize_mode="char", game_state_type=game_state_type,
            #                      token_filter_list=token_filter)
            # result_char, _ = any_model_test(data, dataset_labels, current_game.game_name, scale_mode="row_normalized",
            #                                 feature_names=feature_names,
            #                                 model=model[0][0], model_name=model[0][1], n=n, ax=None, display=False)
            column_name = current_game.game_name + "-" + d_name
            result_dict[column_name] = list()
            r = result_json
            mean = r.mean()
            ste = r.std() / np.sqrt(len(r))
            result_dict[column_name].append(get_CI_str(mean, ste))
            # for r in (result_vec, result_json, result_char):
            #     if r is None:
            #         result_dict[column_name].append(None)
            #         continue
            #     mean = r.mean()
            #     ste = r.std() / np.sqrt(len(r))
            #     result_dict[column_name].append(get_CI_str(mean, ste))
            # for ax in (ax_vec, ax_json):
            #     ax.tick_params(axis='both', which='major', labelsize=15)
            #     if i == len(datasets) - 1:
            #         ax.set_xlabel("Predicted labels", fontdict=label_font, labelpad=10)
            #     else:
            #         ax.set_xlabel("")
            for ax in (ax_vec, ax_json):
                ax.tick_params(axis='both', which='major', labelsize=16)
                if i == 0:
                    ax.set_ylabel("True labels", fontdict=label_font, labelpad=10)
                else:
                    ax.set_ylabel("")
            ax_json.set_xlabel("Predicted labels", fontdict=label_font, labelpad=10)
            ax_vec.set_xlabel("")
            # ax_json.tick_params(axis='both', which='major', labelsize=15)
            # ax_json.set_xlabel("Predicted labels", fontdict=label_font, labelpad=10)
            # ax_json.set_ylabel("True labels", fontdict=label_font, labelpad=10)
            # ax_json.set_title("JSON-Bag", fontsize=20, pad=12)
            i += 1
        if len(datasets) == 1:
            axs[0].set_title("Hand-crafted", fontsize=22, pad=15)
            axs[1].set_title("JSON-Bag", fontsize=22, pad=15)
        else:
            axs[0][int(len(datasets)/2)].set_title("Hand-crafted", fontsize=22, pad=15)
            axs[1][int(len(datasets)/2)].set_title("JSON-Bag", fontsize=22, pad=15)
    plt.savefig(f"{games[0].game_name}-{model[0][1]}.png", dpi=200)
    plt.show()
    # result_pd = pd.DataFrame.from_dict(result_dict)
    # model_name = model[0][1]
    # result_pd = result_pd.rename(
    #     index={0: f"Hand-crafted {model_name}", 1: f"JSON {model_name}", 2: f"JSON Char {model_name}"})
    # result_pd = result_pd.rename(index={0: f"JSON {model_name}"})
    # result_pd.to_csv("Results-L2.csv")
    print(f"FINISHED IN {time.time() - t} SECONDS")
