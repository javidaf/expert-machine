from matplotlib import pyplot as plt
from ml_p2.utils.data_generation import load_classification_data
import numpy as np
from sklearn.metrics import accuracy_score
from ml_p2.visualization.classification_plots import plot_classification_results
from ml_p2.utils.data_generation import create_design_matrix
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.preprocessing import PolynomialFeatures


class LogisticRegression:
    def __init__(
        self,
        learning_rate=0.01,
        max_iterations=1000,
        tol=1e-4,
        lambda_=0.0,
        optimizer="sgd",
        degree=1,
    ):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tol = tol
        self.lambda_ = lambda_
        self.optimizer = optimizer
        self.weights = None
        self.cost_history = []
        self.degree = degree

    def sigmoid(self, z):
        """Compute sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def compute_cost(self, X, y, weights):
        """Compute binary cross-entropy loss with L2 regularization"""
        m = len(y)
        z = X @ weights.reshape(-1, 1)
        h = self.sigmoid(z)
        epsilon = 1e-15  # Prevent log(0)

        # Binary cross-entropy
        cost = -(1 / m) * np.sum(
            y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon)
        )

        # Add L2 regularization if lambda_ > 0
        if self.lambda_ > 0:
            cost += (self.lambda_ / (2 * m)) * np.sum(weights[1:] ** 2)

        return cost

    def compute_gradient(self, X, y, weights):
        """Compute gradient of loss function"""
        m = len(y)
        z = X @ weights.reshape(-1, 1)  # Shape: (m, 1)
        h = self.sigmoid(z)  # Shape: (m, 1)

        # Gradient of binary cross-entropy
        gradient = (1 / m) * X.T @ (h - y)  # Shape: (n_features, 1)

        # Add L2 regularization gradient if lambda_ > 0
        if self.lambda_ > 0:
            reg_term = np.zeros_like(weights).reshape(-1, 1)  # Shape: (n_features, 1)
            reg_term[1:] = weights[1:].reshape(-1, 1)  # Don't regularize bias term
            gradient += (self.lambda_ / m) * reg_term

        return gradient.flatten()

    def fit(self, X, y):
        """Train logistic regression model using gradient descent with polynomial features"""
        X_design = create_design_matrix(X, self.degree)
        m, n = X_design.shape

        if np.all(X_design[:, 0] != 1):
            X_design = np.column_stack([np.ones(m), X_design])

        self.weights = np.zeros(X_design.shape[1])

        prev_cost = float("inf")

        for i in range(self.max_iterations):

            cost = self.compute_cost(X_design, y, self.weights)
            self.cost_history.append(cost)

            if abs(prev_cost - cost) < self.tol:
                break

            gradient = self.compute_gradient(X_design, y, self.weights)
            self.weights -= self.learning_rate * gradient

            prev_cost = cost

        return self

    def predict(self, X):
        """Predict probability of class 1"""
        if self.weights is None:
            raise Exception("Model not trained yet!")

        X_design = create_design_matrix(X, self.degree)

        if np.all(X_design[:, 0] != 1):
            X_design = np.column_stack([np.ones(len(X_design)), X_design])

        return self.sigmoid(X_design @ self.weights.reshape(-1, 1)).flatten()

    def predict_classes(self, X, threshold=0.5):
        """Predict class labels"""
        return (self.predict(X) >= threshold).astype(int)

    def score(self, X, y):
        """Compute accuracy score"""
        return accuracy_score(y.flatten(), self.predict_classes(X))


def grid_search_cv(X, y, param_grid, cv=5):
    """
    Perform grid search with cross-validation for LogisticRegression hyperparameters.

    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Training data
    y : array-like of shape (n_samples,)
        Target values
    param_grid : dict
        Dictionary with parameters names (string) as keys and lists of parameter
        settings to try as values
    cv : int, default=5
        Number of folds for cross-validation

    Returns:
    --------
    best_params : dict
        Dictionary with the best parameters
    best_score : float
        Mean cross-validated score of the best model
    """
    best_score = -np.inf
    best_params = None

    param_combinations = [
        dict(zip(param_grid.keys(), v))
        for v in np.array(np.meshgrid(*param_grid.values())).T.reshape(
            -1, len(param_grid)
        )
    ]

    n_samples = len(X)
    fold_size = n_samples // cv
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    for params in param_combinations:
        scores = []

        for i in range(cv):

            val_indices = indices[i * fold_size : (i + 1) * fold_size]
            train_indices = np.concatenate(
                [indices[: i * fold_size], indices[(i + 1) * fold_size :]]
            )

            X_train_fold = X[train_indices]
            y_train_fold = y[train_indices]
            X_val_fold = X[val_indices]
            y_val_fold = y[val_indices]

            model = LogisticRegression(**params)
            model.fit(X_train_fold, y_train_fold)

            score = model.score(X_val_fold, y_val_fold)
            scores.append(score)

        mean_score = np.mean(scores)

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    return best_params, best_score


class SklearnWrapper:
    """Wrapper class to make sklearn model compatible with our plotting function"""

    def __init__(self, model, poly):
        self.model = model
        self.poly = poly

    def predict(self, X):
        X_poly = self.poly.transform(X)
        return self.model.predict(X_poly)


def run_logistic_regression_comparison(file_path=r"ml-p2\ml_p2\tests\data\chddata.csv"):
    X_train, X_test, y_train, y_test, _, _ = load_classification_data(file_path)
    degree = 1

    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    log_reg = LogisticRegression(learning_rate=0.1, lambda_=0.1, degree=degree)
    log_reg.fit(X_train, y_train)

    sk_log_reg = SklearnLogisticRegression(penalty="l2", C=1 / 0.1)  # C=1/lambda
    sk_log_reg.fit(X_train_poly, y_train)

    sk_wrapper = SklearnWrapper(sk_log_reg, poly)

    print("Performance Metrics:")
    print("-" * 50)
    print(
        f"Custom Logistic Regression Training Score: {log_reg.score(X_train, y_train):.4f}"
    )
    print(
        f"Custom Logistic Regression Test Score:     {log_reg.score(X_test, y_test):.4f}"
    )
    print(
        f"Sklearn Logistic Regression Training Score: {sk_log_reg.score(X_train_poly, y_train):.4f}"
    )
    print(
        f"Sklearn Logistic Regression Test Score:     {sk_log_reg.score(X_test_poly, y_test):.4f}"
    )
    print("-" * 50)

    param_grid = {
        "learning_rate": [0.001, 0.01, 0.1],
        "lambda_": [0.0, 0.1, 1.0],
    }

    best_params, best_score = grid_search_cv(X_train, y_train, param_grid)
    print("\nGrid Search Results:")
    print("-" * 50)
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score: {best_score:.4f}")

    log_reg = LogisticRegression(**best_params)
    log_reg.fit(X_train, y_train)

    plot_classification_results(log_reg, X_test, y_test, "Custom Logistic Regression")
    plot_classification_results(
        sk_wrapper, X_test, y_test, "Sklearn Logistic Regression"
    )


if __name__ == "__main__":
    run_logistic_regression_comparison()
