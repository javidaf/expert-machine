from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from functools import wraps


def add_plot_labels(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title("Cost hisotry vs Iterations")

    return wrapper


@add_plot_labels
def plot_eval_metric_vs_iter(eval_metric: list, params: dict, label: str):
    last_cost = eval_metric[-1]
    last_idx = len(eval_metric) - 1
    """Plot MSE"""
    (line,) = plt.plot(range(len(eval_metric)), eval_metric, label=label)
    plt.scatter(last_idx, last_cost, color=line.get_color(), marker="x", s=40)
    plt.text(
        len(eval_metric),
        eval_metric[-1],
        f"({last_idx}, {last_cost:.3f})",
        verticalalignment="bottom",
        horizontalalignment="center",
    )

    param_text = ", ".join([f"{k}={v}" for k, v in params.items()])
    cell_text = [[k, v] for k, v in params.items()]
    table = plt.table(
        cellText=cell_text,
        colLabels=["Parameter", "Value"],
        loc="upper left",
        cellLoc="center",
        colColours=["lightgrey"] * 2,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(0.4, 1)
    plt.legend()


def plot_surface_3d(x, y, z, title=None, ax=None):
    """
    Plot 3D surface plot.

    Parameters
    ----------
    x : np.ndarray
        x values. Shape (n_grid, n_grid).
    y : np.ndarray
        y values. Shape (n_grid, n_grid).
    z : np.ndarray
        z values. Shape (n_grid, n_grid).
    title : str, optional
        Title of the plot.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new subplot will be created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
def plot_surface_3d(x, y, z, title=None, ax=None):
    """Plot 3D surface plot."""
    if ax is None:
        ax = plt.gca(projection='3d')
    
    surf = ax.plot_surface(x, y, z, cmap=cm.terrain, linewidth=0, antialiased=False)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if title:
        ax.set_title(title)

    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    return ax

