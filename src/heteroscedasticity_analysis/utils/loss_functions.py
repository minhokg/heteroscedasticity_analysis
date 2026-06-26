"""Custom loss functions."""

import tensorflow as tf
from tensorflow import Tensor


def gaussian_nll_loss(
    y_true: Tensor,
    y_pred: Tensor,
) -> Tensor:
    """
    Compute the Gaussian negative log-likelihood (NLL) loss.

    :param y_true: Ground truth values.
    :param y_pred: Model predictions with shape ``(batch_size, 2)``.

    :returns: Scalar Gaussian negative log-likelihood loss.
    """
    mean = y_pred[:, :1]
    log_variance = y_pred[:, 1:]

    variance = tf.exp(log_variance)

    loss = tf.square(y_true - mean) / variance + log_variance

    return tf.reduce_mean(loss)
