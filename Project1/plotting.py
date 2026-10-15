# Imports
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib_params

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
                       s=24, alpha=0.8, label=f"Class{cls}")
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