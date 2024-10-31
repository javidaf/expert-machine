import numpy as np
from sklearn.model_selection import train_test_split
from ml_p2.E41.utils import create_design_matrix


n = 100
x = np.random.rand(n)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.randn(n)
degree = 5
X = create_design_matrix(x[:, np.newaxis], degree=degree)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




print(X.shape)