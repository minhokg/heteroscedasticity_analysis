from collections.abc import Callable

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_regression_model(
    input_dim: int = 1,
    output_dim: int = 2,
    loss: str | Callable | None = None,
) -> Model:
    """
    Build and optionally compile a regression model.

    :param input_dim: Number of input features.
    :param output_dim: Number of outputs.
    :param loss: Loss function. If provided, the model is compiled.

    :returns: Keras regression model.
    """
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(20, activation="tanh"),
            Dense(10, activation="tanh"),
            Dense(output_dim, activation="linear"),
        ]
    )

    if loss is not None:
        model.compile(
            optimizer=Adam(),
            loss=loss,
            metrics=[loss],
        )

    return model
