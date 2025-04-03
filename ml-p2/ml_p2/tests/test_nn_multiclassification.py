from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from ml_p2.neural_network.ffnn import NeuralNetwork
from ml_p2.visualization.classification_plots import (
    plot_decision_boundary_multiclass,
)
import numpy as np

from sklearn.datasets import make_classification


def test_multiclass_classification(n_features=5):
    """
    Test multiclass classification with variable number of features.

    Args:
        n_features (int): Number of features to generate
    """
    # Calculate reasonable defaults for feature types
    n_informative = max(2, int(0.8 * n_features))  # At least 2 informative features
    n_redundant = max(0, n_features - n_informative)  # Rest are redundant

    X, y = make_classification(
        n_samples=300,
        n_features=n_features,
        n_classes=3,
        n_clusters_per_class=1,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    nn = NeuralNetwork(
        input_size=n_features,
        hidden_layers=[32],
        output_size=3,
        hidden_activation="relu",
        output_activation="softmax",
        optimizer="adam",
        learning_rate=0.01,
        classification_type="multiclass",
    )

    onehot = OneHotEncoder(sparse_output=False)
    y_train_onehot = onehot.fit_transform(y_train.reshape(-1, 1))
    y_test_onehot = onehot.transform(y_test.reshape(-1, 1))
    nn.train_classifier(X_train, y_train_onehot, epochs=100)

    train_pred = nn.predict_classes(X_train)
    test_pred = nn.predict_classes(X_test)
    train_acc = np.mean(train_pred == y_train)
    test_acc = np.mean(test_pred == y_test)
    print(f"Number of features: {n_features}")
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    plot_decision_boundary_multiclass(
        nn,
        X_test,
        y_test,
        title=f"Multi-class Classification Results\n(Showing first 2 of {n_features} features)",
    )


if __name__ == "__main__":
    test_multiclass_classification(n_features=5)
    test_multiclass_classification(n_features=10)
    test_multiclass_classification(n_features=3)
