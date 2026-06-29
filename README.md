# Heteroscedasticity Analysis using Neural Networks

A Python implementation demonstrating how neural networks can model **heteroscedastic uncertainty** by comparing a standard regression model trained with Mean Squared Error (MSE) against a probabilistic regression model trained using the Gaussian Negative Log-Likelihood (NLL) loss.

The project generates synthetic heteroscedastic data, trains two neural network models, visualizes their predictions, and compares their predictive performance.

---

## Overview

In many real-world regression problems, the observation noise is **not constant**. Instead, the variance depends on the input features, a phenomenon known as **heteroscedasticity**.

This repository compares two approaches:

- **Standard Neural Network**
  - Predicts only the conditional mean
  - Optimized using Mean Squared Error (MSE)

- **Probabilistic Neural Network**
  - Predicts both the conditional mean and variance
  - Optimized using Gaussian Negative Log-Likelihood (NLL)

The probabilistic model learns not only the expected output but also its uncertainty.


---

## Project Structure

```text
heteroscedasticity_analysis/
│
├── src/
│   └── heteroscedasticity_analysis/
│       │
│       ├── figures
│       │
│       ├── models/
│       │   └── models.py
│       │
│       ├── utils/
│       │   ├── create_plots.py
│       │   ├── evaluation.py
│       │   └── loss_functions.py
│       │
│       └── main.py
│
│
├── requirements.txt
└── README.md
```

---

## Installation


Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

Run the main script

```bash
python src/heteroscedasticity_analysis/main.py
```

---

## Results

The project compares

- Ground truth mean
- Predicted mean
- Ground truth variance
- Predicted variance

for both models.

Evaluation metrics include

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

The Gaussian NLL model is expected to provide more reliable uncertainty estimates while maintaining competitive predictive accuracy.

---

## Example Outputs

The repository generates figures similar to

```
Heteroscedasticity.png
standard_nn.png
nll_nn.png
```

showing

- Synthetic dataset
- Standard neural network predictions
- Gaussian NLL neural network predictions





---

## References

- Nix, D. A., & Weigend, A. S. (1994). *Estimating the Mean and Variance of the Target Probability Distribution.*
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning.*



