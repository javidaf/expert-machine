# import numpy as np
import autograd.numpy as np
from ml_p2.utils import gradient

def Plain_GD(X, y, tol, max_iter, lr=0.1, verbose=True, analytical=True):
    """
    Plain Gradient Descent

    Parameters
    ----------
    X : design matrix
    y : array-like [[y1],[y2],...].T
    lr : float, optional
        learning rate
    n_iter : int, optional
        number of iterations
    """
    beta = np.random.randn(X.shape[1], 1)  # [[b0],[b1]].T
    n = len(y)
    cost_history = []
    cost = float("inf")
    iter = 0

    while cost > tol and iter < max_iter:
        y_pred = X @ beta
        error = y_pred - y
        cost = (1 / n) * np.sum(error**2)
        if not analytical:
            grad = gradient(beta, X, y)
        else:
            grad = (2 / n) * X.T @ error
        beta -= lr * grad
        cost_history.append(cost)
        iter += 1

    if verbose:
        if iter == max_iter:
            print(
                "Did not converge, consider increasing max_iter or increasing learning rate"
            )
        else:
            print(f"Converged in {iter} iterations")

    return cost_history, iter