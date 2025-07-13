import random
from copy import deepcopy

from numpy._typing import NDArray
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from distance_matrix import generate_cross_distance_matrix
from data_vec_utils import FeaturesRangeNormalizer, generate_vec_prototypes
from metrics import jensen_shannon_distance, cosine_distance, euclidean_distance, cosine_dist_vec, euclidean_dist_vec
import numpy as np
from typing import *
from utils import *
from tqdm.auto import tqdm


def label_data(data: List[str], label: str):
    return list((d, label) for d in data)


def knn(dist_matrix: NDArray, y_train: NDArray[int], k: int = 3,
        weighted: Union[bool, Callable[[float], float]] = False):
    pred = np.zeros(dist_matrix.shape[0], dtype=int)
    i = 0
    unique_classes = np.unique(y_train)
    weighted_vote = np.zeros(len(unique_classes))
    for row in dist_matrix:
        sorted_idx = np.argsort(row)
        top_k_idx = sorted_idx[:k]
        top_k_classes = y_train[top_k_idx]
        if weighted is True:
            weighted_distances = 1 / row[top_k_idx]
            for j in range(len(unique_classes)):
                weighted_vote[j] = np.sum(1 / weighted_distances[top_k_classes == j])
            predict_class = weighted_vote.argmax()
        elif type(weighted) is not bool:
            raise NotImplementedError
        else:
            predict_class = max(set(top_k_classes), key=lambda x: (top_k_classes == x).sum())
        pred[i] = predict_class
        i += 1
    return pred


def knn_average(dist_matrix: NDArray, y_train: NDArray[int]):
    pred = np.zeros(dist_matrix.shape[0], dtype=int)
    i = 0
    unique_classes = np.unique(y_train)
    for row in dist_matrix:
        avg_distance = np.zeros(len(unique_classes))
        for c in range(len(unique_classes)):
            avg_distance[c] = np.average(row[y_train == unique_classes[c]])
        pred[i] = avg_distance.argmin()
        i += 1
    return pred


def nn_prototype(x_pred: List[Any], prototypes: List[Any], dist: Callable[[Any, Any], float],
                 decision_margin=False, verbose=False):
    pred = np.zeros(len(x_pred), dtype=int)
    decision_margins = np.zeros(len(x_pred), dtype=np.float32) if decision_margin else None
    class_counts = len(prototypes)
    if verbose:
        print(f"Classifying {class_counts} with prototypes . . .")
    # for i in tqdm(range(len(pred))):
    for i in range(len(pred)):
        distances = np.fromiter((dist(x_pred[i], prototypes[j]) for j in range(class_counts)),
                                dtype=np.float32, count=class_counts)
        pred[i] = np.argmin(distances)
        if decision_margin:
            distances.sort()
            decision_margins[i] = distances[1] - distances[0]
    return pred if not decision_margin else (pred, decision_margins)


def knn_test(data: List[Tuple[Any, int]], dist: Callable[[Any, Any], float],
             seed=None, num_workers=0, weighted=False, k=0, token_filter: List[str] = None,
             train_split=0.25, test_split=0.5, n_shots: int = 0, verbose=False):
    rand = random.Random(seed)
    rand.shuffle(data)
    if n_shots <= 0:
        train_data, test_data = split_data(data, test_size=train_split)
    else:
        train_data, test_data = few_shot_split(data, n_per_class=n_shots)
    if verbose:
        print("Training set size: ", len(train_data))
    x_train, y_train = split_x_y(train_data)
    validation_data, test_data = split_data(test_data, test_size=test_split)
    x_valid, y_valid = split_x_y(validation_data)
    x_test, y_test = split_x_y(test_data)
    if token_filter is not None:
        for x in x_test:
            filter_tokens(x, token_filter)
        for x in x_train:
            filter_tokens(x, token_filter)
        for x in x_valid:
            filter_tokens(x, token_filter)
    if dist is not None:
        dist_matrix = generate_cross_distance_matrix(x_valid, x_train, dist, num_workers=num_workers)
    else:
        raise ValueError("distance function cannot be none!!!")
    acc = list()
    unique_class = set(y_train)
    if verbose:
        print("Training data size: ", len(y_train))
        print("Classifying", len(unique_class), "unique classes on Validation data, size:", len(y_valid))
    if k == 0:
        k_max = min(100, len(train_data))
        # k_max = len(train_data)
        for k in range(1, k_max, 2):
            y_pred = knn(dist_matrix, y_train, k, weighted)
            acc.append(accuracy(y_valid, y_pred))
    else:
        y_pred = knn(dist_matrix, y_train, k, weighted)
        acc.append(accuracy(y_valid, y_pred))
    acc = np.array(acc)
    if verbose:
        print("Highest accuracy is: ", acc.max(), f"at k={(acc.argmax()) * 2 + 1}")
        print("Lowest accuracy is:  ", acc.min(), f"at k={acc.argmin() * 2 + 1}")
        print("Small k accuracy:", acc[:10])

    y_pred = knn(dist_matrix, y_train, k=(acc.argmax()) * 2 + 1, weighted=weighted)
    valid_accuracy = accuracy(y_valid, y_pred, verbose=verbose)
    if verbose:
        print("-------------------------------")
        print("Classifying", len(unique_class), "unique classes on Test data, size:", len(y_test))
        print("Generating cross Distance Matrix for test data ...")
    dist_matrix_test = generate_cross_distance_matrix(x_test, x_train, dist, num_workers=num_workers)
    y_pred = knn(dist_matrix_test, y_train, k=(acc.argmax()) * 2 + 1, weighted=weighted)
    test_accuracy = accuracy(y_test, y_pred, verbose=verbose)
    # print("Validation Accuracy kNN Average:", accuracy(y_test, knn_average(dist_matrix_test, y_train)))
    print("Test Accuracy:", test_accuracy)
    print("--------------------------------------------------------*-----------------------------------------------")
    return valid_accuracy, test_accuracy, metrics.confusion_matrix(y_test, y_pred, normalize="true")


def nn_prototype_test(data, dist: Callable[[Any, Any], float] = jensen_shannon_distance,
                      normalized_mix=False, seed=None, test_size=0.25, n_shots: int = 0,
                      token_filter: List[str] = None,
                      verbose=False):
    rand = random.Random(seed)
    rand.shuffle(data)
    if n_shots <= 0:
        train_data, test_data = split_data(data, test_size=test_size)
    else:
        train_data, test_data = few_shot_split(data, n_per_class=n_shots)
    if verbose:
        print("Training set size: ", len(train_data))
    prototypes = generate_dict_prototypes(train_data, normalized_mix)
    x_test, y_test = split_x_y(test_data)
    # Filter tokens
    if token_filter is not None:
        for x in x_test:
            filter_tokens(x, token_filter)
        for p in prototypes:
            filter_tokens(p, token_filter)
    decision_margins = None
    if verbose:
        y_pred, decision_margins = nn_prototype(x_test, prototypes, dist, decision_margin=verbose, verbose=verbose)
    else:
        y_pred = nn_prototype(x_test, prototypes, dist, verbose=verbose)
    test_accuracy = accuracy(y_test, y_pred, verbose=verbose)
    if verbose:
        print("Test Accuracy:", test_accuracy)
        print("---------------")
        print("Decision Margins Mean: ", decision_margins.mean())
        print("Decision Margins Min : ", decision_margins.min())
        print("Decision Margins Max : ", decision_margins.max())
        print("Decision Margins Std : ", decision_margins.std())
        miss_idx = y_pred != y_test
        if miss_idx.sum() > 0:
            print("Decision Margins Mean of Mis-classifications:", decision_margins[miss_idx].mean())
            print("Decision Margins Min  of Mis-classifications:", decision_margins[miss_idx].min())
            print("Decision Margins Max  of Mis-classifications:", decision_margins[miss_idx].max())
            print("Decision Margins Std  of Mis-classifications:", decision_margins[miss_idx].std())
        print("-------------------------------------------")
    return test_accuracy, metrics.confusion_matrix(y_test, y_pred, normalize="true"), decision_margins


def knn_test_vec(data: np.ndarray, dist: Callable[[Any, Any], float] = euclidean_dist_vec,
                 seed=None, num_workers=0, weighted=False, k=0,
                 train_split=0.25, test_split=0.5, n_shots: int = 0, verbose=False):
    X, y = data[:, :-1], data[:, -1]
    if n_shots > 0:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_shots * len(np.unique(y)), random_state=seed)
        train_index, test_index = list(sss.split(X, y))[0]
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    else:
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=train_split, random_state=seed)
    x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size=test_split, random_state=seed)
    normalizer = FeaturesRangeNormalizer(x_train)
    x_train = normalizer.normalize(x_train)
    x_valid = normalizer.normalize(x_valid)
    x_test = normalizer.normalize(x_test)

    dist_matrix = generate_cross_distance_matrix(x_valid, x_train, dist, num_workers=num_workers)

    acc = list()
    unique_class = set(y_train)
    if verbose:
        print("Training data size: ", len(y_train))
        print("Classifying", len(unique_class), "unique classes on Validation data, size:", len(y_valid))
    if k == 0:
        k_max = min(100, len(x_train))
        # k_max = len(train_data)
        for k in range(1, k_max, 2):
            y_pred = knn(dist_matrix, y_train, k, weighted)
            acc.append(accuracy(y_valid, y_pred))
    else:
        y_pred = knn(dist_matrix, y_train, k, weighted)
        acc.append(accuracy(y_valid, y_pred))
    acc = np.array(acc)
    if verbose:
        print("Highest accuracy is: ", acc.max(), f"at k={(acc.argmax()) * 2 + 1}")
        print("Lowest accuracy is:  ", acc.min(), f"at k={acc.argmin() * 2 + 1}")
        print("Small k accuracy:", acc[:10])

    y_pred = knn(dist_matrix, y_train, k=(acc.argmax()) * 2 + 1, weighted=weighted)
    valid_accuracy = accuracy(y_valid, y_pred, verbose=verbose)
    if verbose:
        # print("Validation Accuracy kNN Average:", accuracy(y_valid, knn_average(dist_matrix, y_train)))
        print("-------------------------------")
        print("Classifying", len(unique_class), "unique classes on Test data, size:", len(y_test))
        print("Generating cross Distance Matrix for test data ...")
    dist_matrix_test = generate_cross_distance_matrix(x_test, x_train, dist, num_workers=num_workers)
    y_pred = knn(dist_matrix_test, y_train, k=(acc.argmax()) * 2 + 1, weighted=weighted)
    test_accuracy = accuracy(y_test, y_pred, verbose=verbose)
    if verbose:
        # print("Test Accuracy kNN Average:", accuracy(y_test, knn_average(dist_matrix_test, y_train)))
        print("Test Accuracy:", test_accuracy)
        print("--------------------------------------------------------*-----------------------------------------------")
    return valid_accuracy, test_accuracy, metrics.confusion_matrix(y_test, y_pred, normalize="true")


def nn_prototype_vec_test(data, dist: Callable[[Any, Any], float] = jensen_shannon_distance,
                          seed=None, test_size=0.25, n_shots: int = 0,
                          verbose=False):
    X, y = data[:, :-1], data[:, -1]
    if n_shots > 0:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_shots * len(np.unique(y)), random_state=seed)
        train_index, test_index = list(sss.split(X, y))[0]
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    else:
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    normalizer = FeaturesRangeNormalizer(x_train)
    x_train = normalizer.normalize(x_train)
    x_test = normalizer.normalize(x_test)
    if verbose:
        print("Training set size: ", len(x_train))
    prototypes = generate_vec_prototypes(x_train, y_train)
    decision_margins = None
    if verbose:
        y_pred, decision_margins = nn_prototype(x_test, prototypes, dist, decision_margin=verbose, verbose=verbose)
    else:
        y_pred = nn_prototype(x_test, prototypes, dist, verbose=verbose)
    test_accuracy = accuracy(y_test, y_pred, verbose)
    if verbose:
        print("Test Accuracy:", test_accuracy)
        print("---------------")
        print("Decision Margins Mean: ", decision_margins.mean())
        print("Decision Margins Min : ", decision_margins.min())
        print("Decision Margins Max : ", decision_margins.max())
        print("Decision Margins Std : ", decision_margins.std())
        miss_idx = y_pred != y_test
        if miss_idx.sum() > 0:
            print("Decision Margins Mean of Mis-classifications:", decision_margins[miss_idx].mean())
            print("Decision Margins Min  of Mis-classifications:", decision_margins[miss_idx].min())
            print("Decision Margins Max  of Mis-classifications:", decision_margins[miss_idx].max())
            print("Decision Margins Std  of Mis-classifications:", decision_margins[miss_idx].std())
        print("-------------------------------------------")
    return test_accuracy, metrics.confusion_matrix(y_test, y_pred, normalize="true"), decision_margins


class NearestPrototypeSearch:
    def __init__(self, dist: Callable[[np.ndarray, np.ndarray], float]):
        self.prototypes: List[np.ndarray] = None
        self.dist = dist

    def fit(self, X_train, y_train):
        self.prototypes = generate_vec_prototypes(X_train, y_train)

    def predict(self, x, verbose=False):
        return nn_prototype(x, self.prototypes, self.dist, verbose=verbose)

