import autograd.numpy as np
from ml_p2.utils import gradient


def adagrad_plain(X, y, tol, max_iter, lr=0.1, verbose=True, analytical=True):

    beta_adagrad_plain = np.random.randn(X.shape[1], 1)  # [[b0],[b1]].T
    epsilon = 1e-8
    G = np.zeros_like(beta_adagrad_plain)
    n = len(y)
    cost_history = []
    iter = 0
    cost = float("inf")
    while cost > tol and iter < max_iter:
        y_pred = X @ beta_adagrad_plain
        error = y_pred - y
        if not analytical:
            grad = gradient(beta_adagrad_plain, X, y)
        else:
            grad = (2 / n) * X.T @ error
        G += grad**2
        adjusted_lr = lr / (np.sqrt(G) + epsilon)
        beta_adagrad_plain -= adjusted_lr * grad
        cost = (1 / n) * np.sum(error**2)
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



def adagrad_momentum_plain(X, y, tol, max_iter, lr, verbose=True, analytical=True):
    """
    Adagrad_momentum_plain

    Parameters
    ----------
    X : design matrix
    y : array-like [[y1],[y2],...].T
    lr : float, optional
        learning rate
    n_iter : int, optional
        number of iterations
    gamma : float, optional
        momentum parameter
    """
    gamma = 0.9
    beta_adagrad_momentum_plain = np.random.randn(X.shape[1], 1)  # [[b0],[b1]].T
    epsilon = 1e-8
    G = np.zeros_like(beta_adagrad_momentum_plain)
    velocity = np.zeros_like(beta_adagrad_momentum_plain)
    n = len(y)
    cost_history = []
    cost = float("inf")
    iter = 0
    while cost > tol and iter < max_iter:
        y_pred = X @ beta_adagrad_momentum_plain
        error = y_pred - y
        if not analytical:
            grad = gradient(beta_adagrad_momentum_plain, X, y)
        else:
            grad = (2 / n) * X.T @ error
        G += grad**2
        adjusted_lr = lr / (np.sqrt(G) + epsilon)
        velocity = gamma * velocity + adjusted_lr * grad
        beta_adagrad_momentum_plain -= velocity
        cost = (1 / n) * np.sum(error**2)
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


def adagrad_sgd(
    X, y, lr=0.1, n_epochs=20, n_batches=10, tol=1e-2, verbose=True, analytical=True
):
    """
    Adagrad with Stochastic Gradient Descent

    Parameters
    ----------
    X : numpy array
        Design matrix.
    y : numpy array
        Target values.
    lr : float, optional
        Initial learning rate.
    n_epochs : int, optional
        Number of epochs.
    n_batches : int, optional
        Number of batches.
    """
    epsilon = 1e-8
    n = len(y)
    batch_size = n // n_batches
    beta = np.random.randn(X.shape[1], 1)
    G = np.zeros_like(beta)
    cost_history = []
    iter = 0
    cost = float("inf")

    for e in range(n_epochs):
        for i in range(0, n, batch_size):
            random_indices = np.random.choice(n, batch_size, replace=False)
            X_batch = X[random_indices]
            y_batch = y[random_indices]

            y_pred = X_batch @ beta
            error = y_pred - y_batch
            if not analytical:
                grad = gradient(beta, X_batch, y_batch)
            else:
                grad = (2 / batch_size) * X_batch.T @ error

            G += grad**2
            adjusted_lr = lr / (np.sqrt(G) + epsilon)
            beta -= adjusted_lr * grad

            cost = (1 / batch_size) * np.sum(error**2)
            cost_history.append(cost)
            iter += 1
            if cost < tol:
                # iter = n_batches * e + i
                if verbose:
                    print(f"Converged in {iter} iterations")
                return cost_history, iter

    if verbose:
        print(
            "Did not converge, consider increasing max_iter or increasing learning rate"
        )

    return cost_history, iter



def adagrad_momentum_sgd(
    X, y, tol, lr=1e-2, n_epochs=20, n_batches=10, verbose=True, analytical=True
):

    gamma = 0.9
    batch_size = len(y) // n_batches
    beta_adagrad_momentum_sgd = np.random.randn(X.shape[1], 1)  # [[b0],[b1]].T
    epsilon = 1e-8
    G = np.zeros_like(beta_adagrad_momentum_sgd)
    velocity = np.zeros_like(beta_adagrad_momentum_sgd)
    cost_history = []
    cost = float("inf")
    iter = 0
    for e in range(n_epochs):
        for i in range(n_batches):
            random_indices = np.random.choice(len(y), batch_size, replace=False)
            X_batch = X[random_indices]
            y_batch = y[random_indices]
            y_pred = X_batch @ beta_adagrad_momentum_sgd
            error = y_pred - y_batch
            if not analytical:
                grad = gradient(beta_adagrad_momentum_sgd, X_batch, y_batch)
            else:
                grad = (2 / batch_size) * X_batch.T @ error
            G += grad**2
            adjusted_lr = lr / (np.sqrt(G) + epsilon)
            velocity = gamma * velocity + adjusted_lr * grad
            beta_adagrad_momentum_sgd -= velocity
            cost = (1 / batch_size) * np.sum(error**2)
            cost_history.append(cost)
            iter += 1
            if cost < tol:
                # iter = n_batches * e + i
                if verbose:
                    print(f"Converged in {iter} iterations")
                return cost_history, iter

    if verbose:
        print(
            "Did not converge, consider increasing max_iter or increasing learning rate"
        )

    return cost_history, iter


