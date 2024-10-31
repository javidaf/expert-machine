import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot as plt


def create_design_matrix(x: np.ndarray, degree: int) -> np.ndarray:
    """
    Create a design matrix with polynomial features.
    ...

    Parameters
    ----------
    x : np.ndarray
        Input data, with shape (n, m), n: #samples and m: #features.
    degree : int
        The degree of the polynomial features.


    Returns
    -------
    np.ndarray
        The return value. True for success, False otherwise.
    """
    poly = PolynomialFeatures(degree=degree, include_bias=True)

    X = poly.fit_transform(x)
    # print("[CREATE DESIGN MATRIX] X shape:", X.shape)
    return X