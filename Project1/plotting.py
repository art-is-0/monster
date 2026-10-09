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

COLORS = {"ME": "lightpink", "LS": "peachpuff", "NN": "powderblue"}

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
    styles = {1: ("o", "tab:red"), 2: ("s", "tab:blue")}

    for ax, (i, j) in zip(axes, pairs):
        for cls, (marker, color) in styles.items():
            mask = y == cls
            ax.scatter(X[mask, i], X[mask, j], marker=marker, color=color,
                       s=18, alpha=0.7, label=f"Class{cls}")
        # ax.set_title(f"Feature {i + 1} vs. Feature {j + 1}")
        ax.set_xlabel(f"Feature {i + 1}")
        ax.set_ylabel(f"Feature {j + 1}")
        ax.grid(True)
        if i == 0 and j == 1:
            ax.legend(loc="lower left")

    # Hide unused axes
    for ax in axes[len(pairs):]:
        ax.set_visible(False)
    # fig.tight_layout()

    fig.savefig(os.path.join(figure_path, f"{name}.pdf"))

    if show:
        plt.show()
    plt.close(fig)


def plot_classifier_comparison(results, name, show=False):
    '''
    One bar panel per dimension d. Each panel shows ME, LS and NN error
    on the best feature combination (by NN error) of that size.
    '''
    n_features = max(len(cols) for cols in results)
    n_panels_per_row = 2
    n_rows = (n_features + n_panels_per_row - 1) // n_panels_per_row

    fig, axes = plt.subplots(n_rows, n_panels_per_row,
                             figsize=(12, 2.5 * n_rows),
                             sharex=True, squeeze=False)
    axes = axes.flatten()

    for ax, d in zip(axes, range(1, n_features + 1)):
        # Best combination of size d, judged by NN error
        combos = [cols for cols in results if len(cols) == d]
        best = min(combos, key=lambda cols: results[cols]["NN"])

        # Classifiers sorted from lowest to highest error
        ranked = sorted(results[best].items(), key=lambda ne: ne[1])
        names = [n for n, _ in ranked]
        errors = [e for _, e in ranked]
        colors = [COLORS[n] for n in names]

        bars = ax.barh(names, errors, color=colors)
        ax.set_title(f"d = {d}, features {", ".join(str(b + 1) for b in best)}")
        ax.set_xlabel("Error rate")
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)

        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
        ax.margins(x=0.1)
    
    # Remove empty panels
    for ax in axes[n_features:]:
        fig.delaxes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(figure_path, f"{name}.pdf"))
    if show:
        plt.show()
    plt.close(fig)