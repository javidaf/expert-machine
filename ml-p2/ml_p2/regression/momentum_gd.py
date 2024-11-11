import autograd.numpy as np
from ml_p2.utils import gradient

def momentum_gd(X, y, tol, max_iter, lr=0.1, gamma=0.9, verbose=True, analytical=True):

    beta_momentum = np.random.randn(X.shape[1], 1)  # [[b0],[b1]].T
    velocity = np.zeros_like(beta_momentum)
    n = len(y)
    cost_history = []

    cost = float("inf")
    iter = 0

    while cost > tol and iter < max_iter:
        y_pred = X @ beta_momentum
        error = y_pred - y
        cost = (1 / n) * np.sum(error**2)
        if not analytical:
            grad = gradient(beta_momentum, X, y)
        else:
            grad = (2 / n) * X.T @ error
        velocity = gamma * velocity + lr * grad
        beta_momentum -= velocity
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


def momentum_sgd(
    X, y, lr=0.1, n_epochs=20, n_batches=10, gamma=0.9, verbose=True, analytical=True
):
    """
    Momentum Stochastic Gradient Descent

    Parameters
    ----------
    X : design matrix
    y : array-like [[y1],[y2],...].T
    lr : float, optional
        learning rate
    n_epochs : int, optional
        number of epochs
    n_batches : int, optional
        number of batches
    gamma : float, optional
        momentum parameter
    """
    n = len(y)
    batch_size = n // n_batches
    beta_momentum_sgd = np.random.randn(X.shape[1], 1)  # [[b0],[b1]].T
    velocity = np.zeros_like(beta_momentum_sgd)
    cost_history = []
    for _ in range(n_epochs):
        for i in range(n_batches):
            random_indices = np.random.choice(n, batch_size, replace=True)
            X_batch = X[random_indices]
            y_batch = y[random_indices]
            y_pred = X_batch @ beta_momentum_sgd
            error = y_pred - y_batch
            if not analytical:
                grad = gradient(beta_momentum_sgd, X_batch, y_batch)
            else:
                grad = (2 / batch_size) * X_batch.T @ error

            velocity = gamma * velocity + lr * grad
            beta_momentum_sgd -= velocity
            cost = (1 / batch_size) * np.sum(error**2)
            cost_history.append(cost)
    return cost_history, beta_momentum_sgd
