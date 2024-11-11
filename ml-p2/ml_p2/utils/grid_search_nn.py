


def grid_search_nn(NeuralNetwork, X_train, X_test, y_train, y_test, param_grid):

    accuracies = []

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for values in param_values[0]:
        for values2 in param_values[1]:
            params = {
                param_names[0]: values,
                param_names[1]: values2
            }
            
            NN = NeuralNetwork(
                input_size=X_test.shape[1],
                hidden_layers=params.get('hidden_layers', [32]),
                output_size=1,
                learning_rate=params.get('learning_rate', 0.01),
                optimizer="adam", 
                output_activation="sigmoid",
                hidden_activation=params.get('hidden_activation', 'leaky_relu'),
                use_regularization=params.get('use_regularization', True),
                lambda_=params.get('lambda_', 0.001),
            )

            NN.train_classifier(X_train, y_train, epochs=100)
            accuracies.append(NN.accuracy_score(X_test, y_test))
            
    return accuracies
