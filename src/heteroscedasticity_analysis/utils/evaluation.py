import logging

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calculate the root mean squared error (RMSE).

    :param y_true: Ground-truth target values.
    :param y_pred: Predicted target values.

    :returns: Root mean squared error.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calculate_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calculate the mean absolute error (MAE).

    :param y_true: Ground-truth target values.
    :param y_pred: Predicted target values.

    :returns: Mean absolute error.
    """
    return float(mean_absolute_error(y_true, y_pred))


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> dict[str, float]:
    """
    Evaluate regression predictions using RMSE and MAE.

    :param y_true: Ground-truth target values.
    :param y_pred: Predicted target values.
    :param model_name: Name of the evaluated model.

    :returns: Dictionary containing the evaluation metrics.
    """
    metrics = {
        "RMSE": calculate_rmse(y_true, y_pred),
        "MAE": calculate_mae(y_true, y_pred),
    }

    logging.info(f"\n{model_name}")
    logging.info("-" * len(model_name))
    for metric_name, value in metrics.items():
        logging.info(f"{metric_name}: {value:.4f}")

    return metrics
