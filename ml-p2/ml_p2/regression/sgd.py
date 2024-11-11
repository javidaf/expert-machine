import autograd.numpy as np
from ml_p2.utils import gradient



def Stochastic_GD(
    X, y, lr=0.1, n_epochs=20, n_batches=10, tol=1e-2, verbose=True, analytical=True
):
    """
    Stochastic Gradient Descent

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
    """

    n = len(y)
    batch_size = n // n_batches
    beta_sgd = np.random.randn(
        X.shape[1],
        1,
    )  # [[b0],[b1]].T

    cost_hostory = []
    iter = float("inf")
    cost = float("inf")
    for e in range(n_epochs):
        for i in range(n_batches):
            random_indices = np.random.choice(len(y), batch_size, replace=False)
            X_batch = X[random_indices]
            y_batch = y[random_indices]
            y_pred = X_batch @ beta_sgd
            error = y_pred - y_batch
            if not analytical:
                grad = gradient(beta_sgd, X_batch, y_batch)
            else:
                grad = (2 / batch_size) * X_batch.T @ error
            beta_sgd -= lr * grad
            cost = (1 / batch_size) * np.sum(error**2)
            cost_hostory.append(cost)
            # learning_rate = learning_rate / (1 + e)
            if cost < tol:
                iter = n_batches * e + i
                if verbose:
                    print(f"Converged in {iter} iterations")
                return cost_hostory, iter

    if verbose:
        print(
            "Did not converge, consider increasing max_iter or increasing learning rate"
        )
    return cost_hostory, iter
