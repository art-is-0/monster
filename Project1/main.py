# Imports
import os
import numpy as np
from dataclasses import dataclass
from classifiers import (
                        minimum_error, minimum_error_predict,
                        least_squares, least_squares_predict,
                        nearest_neighbor, 
                        error_rate
                    )
from plotting import plot_feature_pairs, plot_classifier_comparison
import matplotlib.pyplot as Plot
import itertools
import sys

# ------------------------------------------------------------------------------

# Data loader
@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def load_dataset(path):
    data = np.loadtxt(path)
    y = data[:, 0].astype(int)
    X = data[:, 1:]

    # Odd-numbered -> train, even-numbered -> test
    X_train, y_train = X[0::2], y[0::2]
    X_test, y_test = X[1::2], y[1::2]

    return Dataset(X_train, y_train, X_test, y_test)


def evaluate(ds, cols):
    ''''Error rates (MER, LS) using only the feature columns in cols.'''
    Xtrain = ds.X_train[:, cols]
    Xtest = ds.X_test[:, cols]

    # Minimum Error Rate
    params = minimum_error(Xtrain, ds.y_train)
    mer_err = error_rate(ds.y_test, minimum_error_predict(params, Xtest))

    # Least Squares
    a = least_squares(Xtrain, ds.y_train)
    ls_err = error_rate(ds.y_test, least_squares_predict(a, Xtest))

    # Nearest neighbor
    nn_err = error_rate(ds.y_test, nearest_neighbor(Xtrain, ds.y_train, Xtest))

    return mer_err, ls_err, nn_err

# ------------------------------------------------------------------------------

if __name__ == "__main__":
    filepath = os.path.dirname(os.path.abspath(__file__))
    datasets = {i: load_dataset(os.path.join(filepath, f"datasets/ds-{i}.txt")) for i in (1, 2, 3)}
    results = {}

    for k, ds in datasets.items():
        n_features = ds.X_train.shape[1]
        n_test = len(ds.y_test)
        results[k] = {}

        # Sweep every combination of every size
        for d in range(1, n_features + 1):
            for cols in itertools.combinations(range(n_features), d):
                me_err, ls_err, nn_err = evaluate(ds, list(cols))
                results[k][cols] = {"ME": me_err, "LS": ls_err, "NN": nn_err}

        print(f"Dataset {k}")

        # Rank the feature combinations with nearest neighbor
        print("  Feature combinations ranked by NN error")
        best = {}
        for d in range(1, n_features + 1):
            nn_items = [(cols, errs["NN"]) for cols, errs in results[k].items() if len(cols) == d]
            ranked = sorted(nn_items, key=lambda ce: ce[1])
            best[d] = ranked[0][0]
            print(f"    d = {d}")
            for rank, (cols, err) in enumerate(ranked, start=1):
                print(f"      {rank}. features {", ".join(str(c + 1) for c in cols):<12} "
                      f"error {err:.3f}  ({round(err * n_test)}/{n_test})")
        
        # Rank the classifiers on the best combination for each d
        print()
        print("  Classifiers on the best combination for each d (best first)")
        for d, cols in best.items():
            by_error = sorted(results[k][cols].items(), key=lambda ne: ne[1])
            line = "   ".join(f"{name}: {err:.3f}" for name, err in by_error)
            print(f"   d = {d}  features {", ".join(str(c + 1) for c in cols):<12} {line}")
        print()

        plot_feature_pairs(ds.X_train, ds.y_train, f"dataset_{k}_train")
        plot_classifier_comparison(results[k], f"dataset_{k}_classifiers")