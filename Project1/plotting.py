# Imports
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib_params
from classifiers import (minimum_error, minimum_error_predict,
                         least_squares, least_squares_predict,
                         nearest_neighbor, error_rate)

# ------------------------------------------------------------------------------

filepath = os.path.dirname(os.path.abspath(__file__))
figure_path = os.path.join(filepath, "figures")
os.makedirs(figure_path, exist_ok=True)

# ------------------------------------------------------------------------------

def plot_feature_pairs(X, y, name, show=False):
    '''
    Scatter plots of every pair of features, colored by class.
    Saves the figure to figures/<name>.png
    '''
    n_features = X.shape[1]
    pairs = list(itertools.combinations(range(n_features), 2))

    cols = 3
    rows = (len(pairs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows),
                             squeeze=False)
    axes = axes.flatten()
    styles = {1: ("h", "crimson"), 2: ("p", "cornflowerblue")}

    for ax, (i, j) in zip(axes, pairs):
        for cls, (marker, color) in styles.items():
            mask = y == cls
            ax.scatter(X[mask, i], X[mask, j], marker=marker, color=color,
                       s=24, alpha=0.8, label=f"Class {cls}")
        ax.set_xlabel(f"Feature {i + 1}")
        ax.set_ylabel(f"Feature {j + 1}")
        ax.grid(True, alpha=0.2)
        if i == 0 and j == 1:
            ax.legend(loc="lower left")

    # Hide unused axes
    for ax in axes[len(pairs):]:
        ax.set_visible(False)

    fig.savefig(os.path.join(figure_path, f"{name}.pdf"))

    if show:
        plt.show()
    plt.close(fig)


def plot_decision_regions(ds, cols, name, show=False, n_grid=200):
    '''
    Decision regions of ME, LS, and NN for two chosen features, side by side.
    Classifiers are trained on the training set, test objects are drawn on top
    and the misclassified ones are marked.
    Saves the figure to figures/<name>.pdf
    '''
    cols = list(cols)
    if len(cols) != 2:
        raise ValueError("plot_decision_regions needs exactly two features")
    
    X_train, y_train = ds.X_train[:, cols], ds.y_train
    X_test, y_test = ds.X_test[:, cols], ds.y_test

    # Train on the two chosen features only
    me_params = minimum_error(X_train, y_train)
    ls_a = least_squares(X_train, y_train)
    predictors = {
        "ME": lambda P: minimum_error_predict(me_params, P),
        "LS": lambda P: least_squares_predict(ls_a, P),
        "NN": lambda P: nearest_neighbor(X_train, y_train, P)
    }

    # Grid covering all objects, with a small margin
    X_all = np.vstack([X_train, X_test])
    lo, hi = X_all.min(axis=0), X_all.max(axis=0)
    pad = 0.05 * (hi - lo)
    xx, yy = np.meshgrid(np.linspace(lo[0] - pad[0], hi[0] + pad[0], n_grid),
                            np.linspace(lo[1] - pad[1], hi[1] + pad[1], n_grid))
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    styles = {1: ("h", "crimson"), 2: ("p", "cornflowerblue")}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True,
                                squeeze=False)
    axes = axes.flatten()

    for ax, (clf, predict) in zip(axes, predictors.items()):
        # Label every grid point -> colored regions and the boundary
        Z = np.asarray(predict(grid)).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[0.5, 1.5, 2.5],
                    colors=[styles[1][1], styles[2][1]], alpha=0.15)
        if Z.min() != Z.max():          # Skip if only one class appears
            ax.contour(xx, yy, Z, levels=[1.5], colors="k", linewidths=1)

        # Test objects, with misclassified ones marked
        pred = np.asarray(predict(X_test))
        for cls, (marker, color) in styles.items():
            mask = y_test == cls
            ax.scatter(X_test[mask, 0], X_test[mask, 1], marker=marker,
                        color=color, s=24, alpha=0.8, label=f"Class {cls}")
        wrong = pred != y_test
        ax.scatter(X_test[wrong, 0], X_test[wrong, 1], facecolors="none",
                    edgecolors="k", s=80, linewidths=1, label="Misclassified")

        ax.set_title(f"{clf}, error {error_rate(y_test, pred):.3f}")
        ax.set_xlabel(f"Feature {cols[0] + 1}")
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel(f"Feature {cols[1] + 1}")
    axes[0].legend(loc="lower left")

    fig.savefig(os.path.join(figure_path, f"{name}.pdf"))
    if show:
        plt.show()
    plt.close(fig)