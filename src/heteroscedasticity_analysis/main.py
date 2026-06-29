import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from keras.src.losses import MeanSquaredError

from heteroscedasticity_analysis.models.models import build_regression_model
from heteroscedasticity_analysis.utils.create_plots import plot_dataset, plot_predictions
from heteroscedasticity_analysis.utils.evaluation import evaluate_predictions
from heteroscedasticity_analysis.utils.loss_functions import gaussian_nll_loss

if __name__ == "__main__":
    # generate n sample data
    np.random.seed(0)
    n = 2000
    x = np.linspace(0, np.pi / 2, n)
    mx = np.sin(4 * x)
    fx = mx * np.sin(5 * x)  # function part of x
    sigma2 = 0.02 + 0.02 * (1 - mx) ** 2
    error = np.random.normal(0, np.sqrt(sigma2), n)
    y = fx + error

    # Plotting x vs y and x vs sigma square (the variance of error)
    plot_dataset(x=x, true_function=fx, observations=y, variance=sigma2, save_path="analysis/Heteroscedatsticity.png")

    # To do a prediction, we need to divide data into train and test
    ntest = int(n / 4)
    test_idx = np.random.choice(n, ntest, replace=False)
    train_idx = np.setdiff1d(np.arange(n), test_idx)
    ntrain = train_idx.shape[0]

    xr = x[train_idx]
    xt = x[test_idx]
    fxr = fx[train_idx]
    fxt = fx[test_idx]
    sigma2r = sigma2[train_idx]
    sigma2t = sigma2[test_idx]
    yr = y[train_idx].reshape(ntrain, 1)
    yt = y[test_idx].reshape(ntest, 1)

    model = build_regression_model(loss=MeanSquaredError())

    # Graph for test data in a standard Neural Network
    yth_standard = model.predict(xt)
    sig2th_standard = yth_standard[:, 1]

    plot_predictions(
        x=xt,
        true_mean=fxt,
        predicted_mean=yth_standard[:, 0],
        true_variance=sigma2t,
        predicted_variance=yth_standard[:, 1],
        title="Standard Neural Network",
        save_path="analysis/standard_nn.png",
    )
    model_nll = build_regression_model(loss=gaussian_nll_loss)
    yth_nll = model_nll.predict(xt)
    sig2th_nll = np.exp(yth_nll[:, 1])

    plot_predictions(
        x=xt,
        true_mean=fxt,
        predicted_mean=yth_nll[:, 0],
        true_variance=sigma2t,
        predicted_variance=np.exp(yth_nll[:, 1]),
        title="Negative Log-Liklihood Neural Network",
        save_path="analysis/nll_nn.png",
    )

    # Comparing the performance between MSE CNN and NLL CNN by using RMSE (Root MSE) and MAE
    # RMSE
    evaluate_predictions(y_true=fxt, y_pred=yth_standard[:, 0], model_name="Standard Neural Network")

    # MAE
    evaluate_predictions(
        y_true=fxt,
        y_pred=yth_nll[:, 0],
        model_name="Negative Log-Likelihood Neural Network",
    )
