import matplotlib.pyplot as plt

# import seaborn as sns
# # Plot settings
# # sns.color_palette('bright')
# sns.set_theme()
# # sns.set_style('darkgrid')
# sns.set_theme(style="whitegrid")
# style and parameters used for plotting
# plt.style.use("bmh")
pad = 8 / 72 
font_size = 16

plt.rcParams.update(
    {
        # Figure
        "figure.dpi": 600,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.w_pad": pad,
        "figure.constrained_layout.h_pad": pad,
        "savefig.format": "pdf",
        # Plotting
        "lines.linewidth": 2,
        # Axes
        "axes.linewidth": 0.5,
        "axes.grid": True,
        "grid.color": "black",
        "grid.alpha": 0.25,
        "axes.labelpad": 2.0,
        "axes.titlepad": 5.0,
        "ytick.major.width": 0.6,
        "xtick.major.width": 0.6,
        "ytick.minor.width": 0.3,
        "xtick.minor.width": 0.3,
        # Fonts
        "legend.fontsize": font_size,
        "axes.titlesize": font_size,
        "figure.titlesize": font_size + 1,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 2,
        "ytick.labelsize": font_size - 2,
        # Legend
        "legend.handletextpad": 0.3,
        "legend.scatterpoints": 3,
        "legend.borderaxespad": 0.1,
        "legend.fancybox": False,
        "patch.linewidth": 0.5,
        "legend.columnspacing": 1,
        # "legend.edgecolor": "white",
    }
)