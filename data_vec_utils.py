import numpy as np
import json
import pandas as pd
from typing import *
from sklearn.linear_model import LinearRegression


def process_score_vector(score_vector, n_players):
    result = list()
    result.extend(score_vector[-n_players:])    # append first the final score
    x_train = np.arange(1, len(score_vector)/n_players + 1).reshape(-1, 1)
    # append the intercept and slope of each player score progression
    for i in range(n_players):
        y_train = score_vector[i::n_players]
        m = LinearRegression().fit(x_train, y_train)
        result.append(m.intercept_)
        result.append(m.coef_[0])
    # result.extend(score_vector)     # append the entire score vector
    return result


def process_data(csv_path: str, features, n_players, label=-1):
    df = pd.read_csv(csv_path)
    processed_data = list()
    for i in range(len(df[features].values)):
        d = list(df[features].values[i])
        score_vec = None
        if "ScoresPerRound(ScoresPerRound)" in df.columns:
            score_vec = json.loads(df['ScoresPerRound(ScoresPerRound)'].values[i])
        elif "ScoresPerTurn(ScoresPerTurn)" in df.columns:
            score_vec = json.loads(df['ScoresPerTurn(ScoresPerTurn)'].values[i])
        if score_vec is not None:
            d.extend(process_score_vector(score_vec, n_players=n_players))
        if label >= 0:
            d.append(label)
        processed_data.append(np.array(d))
    return np.array(processed_data)


def generate_vec_prototypes(X: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
    labels = set(y)
    prototypes = [np.zeros(0)] * len(labels)
    for i in labels:
        prototypes[int(i)] = (X[y == i]).mean(axis=0)
    return prototypes


class FeaturesRangeNormalizer:
    def __init__(self, data: np.ndarray):
        self.min_features = np.min(data, axis=0)
        self.max_features = np.max(data, axis=0)
        self.range = self.max_features-self.min_features
        self.range[self.range == 0] = 1

    def normalize(self, data: np.ndarray):
        return (data - self.min_features)/self.range

