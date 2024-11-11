# import numpy as np
import autograd.numpy as np
from ml_p2.utils import gradient


def adam_plain(
    X, y, tol, max_iter, lr=0.001, beta1=0.9, beta2=0.999, verbose=True, analytical=True
):
    """
    Adam optimizer for plain Gradient Descent

    Parameters
    ----------
    X : design matrix
    y : array-like [[y1],[y2],...].T
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations
    lr : float, optional
        Learning rate
    verbose : bool, optional
        If True, print convergence information
    """
    epsilon = 1e-8
    """
    beta1 :
    Exponential decay rate for the first moment estimates
    beta2 :
    Exponential decay rate for the second moment estimates
    """
    beta = np.random.randn(X.shape[1], 1)
    m = np.zeros_like(beta)
    v = np.zeros_like(beta)
    n = len(y)
    cost_history = []
    iter = 0
    cost = float("inf")

    while cost > tol and iter < max_iter:
        y_pred = X @ beta
        error = y_pred - y
        cost = (1 / n) * np.sum(error**2)
        if not analytical:
            grad = gradient(beta, X, y)
        else:
            grad = (2 / n) * X.T @ error
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)
        m_hat = m / (1 - beta1 ** (iter + 1))
        v_hat = v / (1 - beta2 ** (iter + 1))
        beta -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
        cost_history.append(cost)
        iter += 1

    if verbose:
        if iter == max_iter:
            print(
                "Did not converge, consider increasing max_iter or adjusting learning rate"
            )
        else:
            print(f"Converged in {iter} iterations")

    return cost_history, iter



def adam_sgd(
    X,
    y,
    lr=0.001,
    n_epochs=20,
    n_batches=10,
    beta1=0.9,
    beta2=0.999,
    tol=1e-4,
    verbose=True,
    analytical=True,
):
    """
    Adam optimizer for Stochastic Gradient Descent

    Parameters
    ----------
    X : design matrix
    y : array-like [[y1],[y2],...].T
    lr : float, optional
        Learning rate
    n_epochs : int, optional
        Number of epochs
    n_batches : int, optional
        Number of batches
    beta1 : float, optional
        Exponential decay rate for the first moment estimates
    beta2 : float, optional
        Exponential decay rate for the second moment estimates
    tol : float, optional
        Tolerance for convergence
    verbose : bool, optional
        If True, print convergence information
    """
    epsilon = 1e-8
    n = len(y)
    batch_size = n // n_batches
    beta = np.random.randn(X.shape[1], 1)
    m = np.zeros_like(beta)
    v = np.zeros_like(beta)
    cost_history = []
    iter = 0
    cost = float("inf")
    t = 0

    for e in range(n_epochs):
        for i in range(n_batches):
            t += 1
            random_indices = np.random.choice(n, batch_size, replace=False)
            X_batch = X[random_indices]
            y_batch = y[random_indices]
            y_pred = X_batch @ beta
            error = y_pred - y_batch
            if analytical:
                grad = (2 / batch_size) * X_batch.T @ error
            else:
                grad = gradient(beta, X_batch, y_batch)
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad**2)
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            beta -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
            cost = (1 / batch_size) * np.sum(error**2)
            cost_history.append(cost)
            iter += 1
            if cost < tol:
                if verbose:
                    print(f"Converged in {iter} iterations")
                return cost_history, iter

    if verbose:
        print(
            "Did not converge, consider increasing n_epochs or adjusting learning rate"
        )

    return cost_history, iter

