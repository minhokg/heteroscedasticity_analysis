"""Utility functions for visualizing heteroscedastic regression results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_dataset(
    x: np.ndarray,
    true_function: np.ndarray,
    observations: np.ndarray,
    variance: np.ndarray,
    save_path: str | Path | None = None,
) -> Figure:
    """
    Plot the generated dataset and the heteroscedastic variance.

    :param x: Input values
    :param true_function: Noise-free target values.
    :param observations: Observed target values with noise.
    :param variance: True variance associated with each observation.
    :param save_path: File path where the figure will be saved. If ``None``, the figure is not saved.

    :returns: The created Matplotlib figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))

    axes[0].set_title("Heteroscedasticity")
    axes[0].plot(x, true_function, "r.", label="True function")
    axes[0].plot(x, observations, "k.", label="Observations")
    axes[0].set_ylabel("y")
    axes[0].legend()

    axes[1].plot(x, variance, "r.")
    axes[1].set_ylabel(r"$\sigma^2$")
    axes[1].set_xlabel("x")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_predictions(
    x: np.ndarray,
    true_mean: np.ndarray,
    predicted_mean: np.ndarray,
    true_variance: np.ndarray,
    predicted_variance: np.ndarray,
    title: str,
    save_path: str | Path | None = None,
) -> Figure:
    """
    Plot predicted mean and predicted variance.

    :param x: Input values.
    :param true_mean: Ground-truth target values.
    :param predicted_mean: Predicted target values.
    :param true_variance: Ground-truth variance.
    :param predicted_variance: Predicted variance.
    :param title: Title of the figure.
    :param save_path: File path where the figure will be saved. If ``None``, the figure is not saved.
    :returns: Figure

    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    axes[0].set_title(title)
    axes[0].plot(x, true_mean, "r.", label="True")
    axes[0].plot(x, predicted_mean, "b.", label="Prediction")
    axes[0].set_ylabel("y")
    axes[0].legend()

    axes[1].plot(x, true_variance, "r.", label="True")
    axes[1].plot(x, predicted_variance, "b.", label="Prediction")
    axes[1].set_ylabel(r"$\sigma^2$")
    axes[1].set_xlabel("x")
    axes[1].legend()

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    return fig
