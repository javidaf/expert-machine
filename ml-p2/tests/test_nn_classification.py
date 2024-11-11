from ml_p2.neural_network.ffnn import NeuralNetwork
from ml_p2.utils.data_generation import load_classification_data
from ml_p2.visualization.classification_plots import plot_classification_results, plot_decision_boundary2
import numpy as np
from sklearn.neural_network import MLPClassifier
from ml_p2.visualization.classification_plots import plot_data


def test_classification(file_path, hidden_layers=[32], output_size=1, hidden_activation='relu',
                       output_activation='sigmoid', optimizer='adam', learning_rate=0.01,
                       use_regularization=False, lambda_=0.01, initializer='he'):
    X_train, X_test, y_train, y_test, X_train_minmax, X_test_minmax = load_classification_data(file_path)
    full_data_features = np.concatenate((X_train, X_test), axis=0) 
    full_data_labels = np.concatenate((y_train, y_test), axis=0)    
    plot_data(full_data_features, full_data_labels)

    # X_train = X_train_minmax
    # X_test = X_test_minmax
    
    nn = NeuralNetwork(
        input_size=X_train.shape[1],
        hidden_layers=hidden_layers,
        output_size=output_size,
        hidden_activation=hidden_activation,
        output_activation=output_activation,
        optimizer=optimizer,
        learning_rate=learning_rate,
        use_regularization=use_regularization,
        lambda_=lambda_,
        initializer=initializer
    )
    sklearn_nn = MLPClassifier(
    hidden_layer_sizes=hidden_layers,
    activation='relu',
        solver='adam',
        learning_rate_init=learning_rate,
        max_iter=500,
        random_state=42
    )

    sklearn_nn.fit(X_train, y_train.ravel())
    sklearn_accuracy = sklearn_nn.score(X_test, y_test)
   
    nn.train_classifier(X_train, y_train, epochs=500, batch_size=10)
    plot_classification_results(nn, X_train, y_train, title="Training Data Classification")
    plot_classification_results(nn, X_test, y_test, title="Test Data Classification")
    plot_decision_boundary2(sklearn_nn, X_test, y_test)


    train_acc = nn.accuracy_score(X_train, y_train)
    test_acc = nn.accuracy_score(X_test, y_test)
    print(f"Custom NN Training Accuracy: {train_acc:.4f}")
    print(f"Custom NN Test Accuracy: {test_acc:.4f}")
    print(f"Scikit-learn MLPClassifier Test Accuracy: {sklearn_accuracy:.4f}")

if __name__ == "__main__":
    test_classification(r'ml-p2\tests\data\chddata.csv')

