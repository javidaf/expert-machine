import numpy as np
import pandas as pd
import rasterio
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

def FrankeFunction(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Franke's test function.
    ...

    Parameters
    ----------
    x : np.ndarray
        meshgrid x values.
    y : np.ndarray
        meshgrid y values.

    Returns
    -------
    np.ndarray
        meshgrid z values.
    """
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4


def simple_1d_function(x):
    return 2 + 3 * x - 4 * x**2 + 5 * x**3


def Noise(s, n):
    return np.random.normal(0, s, n)

def generate_data(n_samples):
    """Return data in the shape"""
    x = np.linspace(0, 1, n_samples)
    y = 1 + 2 * x + 3 * x**2 + np.random.normal(0, 0.1, n_samples)
    return x, y


def create_design_matrix(x: np.ndarray, degree: int) -> np.ndarray:
    """
    Create a design matrix with polynomial features.
    ...

    Parameters
    ----------
    x : np.ndarray
        Input data, with shape (n, m), n: #samples and m: #features.
    degree : int
        The degree of the polynomial features.


    Returns
    -------
    np.ndarray
        The return value. True for success, False otherwise.
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)

    X = poly.fit_transform(x)
    # print("[CREATE DESIGN MATRIX] X shape:", X.shape)
    return X


def generate_ff_data(n: int = 20, noise: bool = False):
    """
    Generate data for Franke's function.
    ...

    Parameters
    ----------
    n : int
        Number of samples.
    noise : bool, optional
        Add noise to the data, by default False.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        xy = [[x1  y1] [x2  y2] [x3  y3] ...] shape (n, 2).
        z = meshgrid z values.
    """
    # x = np.arange(0, 1, 1 / n)
    # y = np.arange(0, 1, 1 / n)
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    # xy = np.concatenate(
    #     (x.reshape(-1, 1), y.reshape(-1, 1)), axis=1
    # )  # [[x1  y1] [x2  y2] [x3  y3] ...]

    x_mesh, y_mesh = np.meshgrid(x, y)
    xy = np.column_stack([x_mesh.ravel(), y_mesh.ravel()])  # Shape: (n*n, 2)
    if noise:
        z = FrankeFunction(x_mesh, y_mesh) + Noise(0.1, x_mesh.shape)
    else:
        z = FrankeFunction(x_mesh, y_mesh)

    z = z.ravel()
    # print("[GENERATE FF DATA] xy shape:", xy.shape)
    # print("[GENERATE FF DATA] z shape:", z.shape)

    return xy, z


def read_and_preprocess_terrain(tif_file, subset_rows=None, subset_cols=None):
    with rasterio.open(tif_file) as dataset:
        z = dataset.read(1)
        transform = dataset.transform

    n_rows, n_cols = z.shape

    # Determine subset size
    if subset_rows is None or subset_cols is None:
        subset_rows, subset_cols = n_rows, n_cols

    # Calculate starting indices for the subset
    start_row = (n_rows - subset_rows) // 2
    start_col = (n_cols - subset_cols) // 2

    # Extract the subset
    z_subset = z[
        start_row : start_row + subset_rows, start_col : start_col + subset_cols
    ]

    rows, cols = np.meshgrid(
        np.arange(start_row, start_row + subset_rows),
        np.arange(start_col, start_col + subset_cols),
        indexing="ij",
    )
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    xs_flat = np.array(xs).flatten()
    ys_flat = np.array(ys).flatten()
    z_flat = z_subset.flatten()

    # Remove invalid data points
    valid_mask = ~np.isnan(z_flat)
    xs_flat = xs_flat[valid_mask]
    ys_flat = ys_flat[valid_mask]
    z_flat = z_flat[valid_mask]

    xy = np.column_stack((xs_flat, ys_flat))
    return xy, z_flat, valid_mask, z.shape


def load_classification_data(filepath, test_size=0.2, random_state=4):
    """
    Load and process classification data from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    test_size : float, default=0.2
        Proportion of dataset to include in the test split
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    X_train : array
        Training features (scaled)
    X_test : array
        Test features (scaled)
    y_train : array
        Training labels
    y_test : array
        Test labels
    """
    data = pd.read_csv(filepath, header=None)
    
    X = data.iloc[:, 1:-1].values  # All columns except the first and last
    y = data.iloc[:, -1].values    # Last column
    

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    min_max_scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_minmax = min_max_scaler.fit_transform(X_train)
    X_test_minmax = min_max_scaler.transform(X_test)
    
    # Reshape labels for neural network
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X_train_minmax, X_test_minmax