import autograd.numpy as np
from ml_p2.utils import gradient

def rmsprop_plain(X, y, tol, max_iter, lr=1e-2, verbose=True, analytical=True):
    epsilon = 1e-8
    rho = 0.9
    beta_rmsprop = np.random.randn(X.shape[1], 1)
    E_g2 = np.zeros_like(beta_rmsprop)
    n = len(y)
    cost_history = []
    iter = 0
    cost = float("inf")

    while cost > tol and iter < max_iter:
        y_pred = X @ beta_rmsprop
        error = y_pred - y
        if not analytical:
            grad = gradient(beta_rmsprop, X, y)
        else:
            grad = (2 / n) * X.T @ error
        E_g2 = rho * E_g2 + (1 - rho) * grad**2
        adjusted_lr = lr / (np.sqrt(E_g2) + epsilon)
        beta_rmsprop -= adjusted_lr * grad
        cost = (1 / n) * np.sum(error**2)
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


def rmsprop_sgd(
    X,
    y,
    lr=0.01,
    n_epochs=20,
    n_batches=10,
    beta=0.9,
    tol=1e-4,
    verbose=True,
    analytical=True,
):
    epsilon = 1e-8
    beta_rmsprop_sgd = np.random.randn(X.shape[1], 1)
    n = len(y)
    batch_size = n // n_batches
    E_g2 = np.zeros_like(beta_rmsprop_sgd)
    cost_history = []
    iter = 0
    cost = float("inf")

    for e in range(n_epochs):
        for _ in range(n_batches):
            random_indices = np.random.choice(n, batch_size, replace=False)
            X_batch = X[random_indices]
            y_batch = y[random_indices]
            y_pred = X_batch @ beta_rmsprop_sgd
            error = y_pred - y_batch
            if not analytical:
                grad = gradient(beta_rmsprop_sgd, X_batch, y_batch)
            else:
                grad = (2 / batch_size) * X_batch.T @ error
            E_g2 = beta * E_g2 + (1 - beta) * grad**2
            adjusted_lr = lr / (np.sqrt(E_g2) + epsilon)
            beta_rmsprop_sgd -= adjusted_lr * grad
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
