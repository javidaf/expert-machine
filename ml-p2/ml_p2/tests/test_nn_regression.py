from matplotlib import pyplot as plt
from ml_p2.neural_network.ffnn import NeuralNetwork
from ml_p2.utils.data_generation import read_and_preprocess_terrain
from ml_p2.visualization.plotting import plot_eval_metric_vs_iter, plot_surface_3d
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from ml_p2.utils.data_generation import generate_ff_data, FrankeFunction
import os


def terrain_test():
    script_dir = os.path.dirname(__file__)

    tif_file = os.path.join(script_dir, "data", "SRTM_data_Norway_1.tif")

    tif_file = os.path.normpath(tif_file)
    # tif_file = r"ml-p2\tests\data\SRTM_data_Norway_1.tif"
    subset_size = 100
    X, y, valid_mask, original_shape = read_and_preprocess_terrain(
        tif_file, subset_size, subset_size
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    # Scale y values between 0 and 1 since we're using sigmoid activation
    scaler_y = MinMaxScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test = scaler_y.transform(y_test.reshape(-1, 1))

    nn = NeuralNetwork(
        input_size=X_train.shape[1],
        hidden_layers=[32],
        output_size=1,
        hidden_activation="sigmoid",  # ReLU might work better for terrain
        output_activation="sigmoid",
        optimizer="adam",
        learning_rate=0.01,
    )

    nn.train(X_train, y_train, epochs=50, batch_size=32)

    params = nn.params
    plt.figure(figsize=(8, 6))
    plot_eval_metric_vs_iter(nn.cost_history, params, "Training Cost")
    plt.show()
    plt.close()

    y_pred = nn.predict(X_test)
    y_pred_original = scaler_y.inverse_transform(y_pred)
    y_test_original = scaler_y.inverse_transform(y_test)

    mse = mean_squared_error(y_test_original, y_pred_original)
    r2 = nn.score(X_test, y_test)
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R2 Score: {r2:.4f}")

    x_unique = np.unique(X[:, 0])
    y_unique = np.unique(X[:, 1])
    x_mesh, y_mesh = np.meshgrid(
        x_unique,
        y_unique,
    )

    X_grid = np.column_stack([x_mesh.ravel(), y_mesh.ravel()])
    X_grid_scaled = scaler_X.transform(X_grid)

    z_pred = nn.predict(X_grid_scaled)
    z_pred = scaler_y.inverse_transform(z_pred).reshape(x_mesh.shape)

    z_true = y.reshape(x_mesh.shape)

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(121, projection="3d")
    plot_surface_3d(x_mesh, y_mesh, z_true, title="Original Terrain", ax=ax1)

    ax2 = fig.add_subplot(122, projection="3d")
    plot_surface_3d(
        x_mesh,
        y_mesh,
        np.flip(np.rot90(z_pred, k=2), axis=1),
        title="Predicted Terrain",
        ax=ax2,
    )

    plt.show()
    plt.close()


def NNRegression():
    grid_size = 50
    x_grid = np.linspace(0, 1, grid_size)
    y_grid = np.linspace(0, 1, grid_size)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    X_grid = np.column_stack([x_mesh.ravel(), y_mesh.ravel()])

    X, y = generate_ff_data(n=100, noise=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    X_grid = scaler_X.transform(X_grid)

    # Scale y values between 0 and 1
    scaler_y = MinMaxScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test = scaler_y.transform(y_test.reshape(-1, 1))

    nn = NeuralNetwork(
        input_size=X_train.shape[1],
        hidden_layers=[32],
        output_size=1,
        hidden_activation="sigmoid",  # or 'relu', 'sigmoid', 'tanh'
        output_activation="sigmoid",
        optimizer="adam",  # or 'sgd', 'rmsprop', 'adagrad'
        learning_rate=0.01,
        use_regularization=False,
        lambda_=0.01,
    )

    nn.train(X_train, y_train, epochs=200)
    params = nn.params
    plot_eval_metric_vs_iter(nn.cost_history, params, "Training Cost")

    y_pred = nn.predict(X_test)
    y_pred_train = nn.predict(X_train)

    y_pred_original = scaler_y.inverse_transform(y_pred)
    y_test_original = scaler_y.inverse_transform(y_test)
    y_train_original = scaler_y.inverse_transform(y_train)
    loss_test = mean_squared_error(y_test, y_pred)
    loss_train = mean_squared_error(y_train, y_pred_train)
    print(f"Test MSE: {loss_test}")
    print(f"Train MSE: {loss_train}")

    y_true_grid = FrankeFunction(x_mesh, y_mesh)
    y_pred_grid = nn.predict(X_grid).reshape(x_mesh.shape)
    y_pred_grid = scaler_y.inverse_transform(y_pred_grid.reshape(-1, 1)).reshape(
        x_mesh.shape
    )

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(121, projection="3d")
    plot_surface_3d(
        x_mesh, y_mesh, y_true_grid, title="Original Franke Function", ax=ax1
    )

    ax2 = fig.add_subplot(122, projection="3d")
    plot_surface_3d(
        x_mesh, y_mesh, y_pred_grid, title="Predicted Franke Function", ax=ax2
    )

    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    terrain_test()
    NNRegression()
