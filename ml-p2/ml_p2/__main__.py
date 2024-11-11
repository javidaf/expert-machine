import numpy as np
from sklearn.model_selection import train_test_split
from ml_p2.utils.data_generation import (
    FrankeFunction,
    Noise,
    generate_ff_data,
    create_design_matrix,
)

from ml_p2.regression.ols import OLS 

# X, y = generate_ff_data(n_samples=100, noise=True)


# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
