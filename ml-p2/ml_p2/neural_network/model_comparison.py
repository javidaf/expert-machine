import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from ml_p2.utils.data_generation import generate_ff_data
from ml_p2.neural_network.ffnn import NeuralNetwork
from ml_p2.utils.data_generation import create_design_matrix

def compare_models(X, y, test_size=0.2, random_state=42):
    """Compare different models on the same dataset."""
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train_poly = create_design_matrix(X_train, degree=5)
    X_test_poly = create_design_matrix(X_test, degree=5)
    
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    X_scaler = StandardScaler()
    X_poly_scaler = StandardScaler()
    y_scaler = MinMaxScaler()
    
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    X_train_poly_scaled = X_poly_scaler.fit_transform(X_train_poly)
    X_test_poly_scaled = X_poly_scaler.transform(X_test_poly)
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)
    
    # Initialize models
    models = {
        'Custom NN': NeuralNetwork(
            input_size=X_train.shape[1],
            hidden_layers=[32],
            output_size=1,
            hidden_activation='sigmoid',
            output_activation='sigmoid',
            optimizer='adam',
            learning_rate=0.01
        ),
        'SKLearn NN': MLPRegressor(
            hidden_layer_sizes=(32),
            activation='relu',
            max_iter=2000,
            learning_rate_init=0.01,
            random_state=random_state
        ),
        'Linear Regression (Poly)': LinearRegression(),
        'Ridge Regression (Poly)': Ridge(alpha=0.1, random_state=42),
    }
    
    results = {}
    
    for name, model in models.items():
        if name == 'Custom NN':
            model.train(X_train_scaled, y_train_scaled, epochs=200)
            y_pred_scaled = model.predict(X_test_scaled)
        elif name in ['Linear Regression (Poly)', 'Ridge Regression (Poly)']:
            # Use scaled polynomial features for Linear and Ridge regression
            model.fit(X_train_poly_scaled, y_train_scaled.ravel())
            y_pred_scaled = model.predict(X_test_poly_scaled).reshape(-1, 1)
        else:
            model.fit(X_train_scaled, y_train_scaled.ravel())
            y_pred_scaled = model.predict(X_test_scaled).reshape(-1, 1)
        
        # Transform predictions back to original scale
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        y_test_orig = y_scaler.inverse_transform(y_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_scaled, y_pred_scaled)
        r2 = r2_score(y_test_scaled, y_pred_scaled)
        
        results[name] = {
            'MSE': mse,
            'R2': r2
        }
    
    return results

def main():
    # Generate data
    X, y = generate_ff_data(n=100, noise=True)
    
    # Compare models
    results = compare_models(X, y)
    
    # Print results
    print("\nModel Comparison Results:")
    print("-" * 50)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"MSE: {metrics['MSE']:.6f}")
        print(f"R2 Score: {metrics['R2']:.6f}")

if __name__ == "__main__":
    main() 