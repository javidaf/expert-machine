
# import numpy as np
import autograd.numpy as np
from autograd import grad


def R2_score(y_true, y_pred):
    """Return R2 score"""
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - ss_res / ss_tot


def MSE(y_true, y_pred):
    """Return Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)


def custom_mean(x):
    return np.sum(x) / x.size


def loss_function(beta, X, y):
    y_pred = X @ beta
    error = y_pred - y
    return custom_mean(error**2)


def ridge_loss_function(beta, X, y, lambda_):
    y_pred = X @ beta
    error = y_pred - y
    return custom_mean(error**2) + lambda_ * np.sum(beta**2)


gradient = grad(loss_function)
gradient_ridge = grad(ridge_loss_function)