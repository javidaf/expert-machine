import numpy as np
import matplotlib.pyplot as plt


def generate_data(n_samples):
    x = np.linspace(0, 1, n_samples)
    y = 1 + 2 * x + 3 * x**2 + np.random.normal(0, 0.1, n_samples)  # a0=1, a1=2, a2=3
    return x, y


x, y = generate_data(100)
X = np.column_stack((np.ones(len(x)), x, x**2))
y = y.reshape(-1, 1)

learning_rate = 0.1
n_iterations = 50

beta = np.random.randn(X.shape[1], 1)
cost_history = []

# Plain Gradient Descent
for i in range(n_iterations):
    y_pred = X @ beta
    error = y_pred - y
    cost = (1 / len(y)) * np.sum(error**2)
    cost_history.append(cost)
    gradient = (2 / len(y)) * X.T @ error
    beta -= learning_rate * gradient

beta_momentum = np.random.randn(X.shape[1], 1)
velocity = np.zeros_like(beta_momentum)
beta_momentum = np.random.randn(X.shape[1], 1)
velocity = np.zeros_like(beta_momentum)
gamma = 0.9  # Momentum coefficient
cost_history_momentum = []

# Gradient descent with Momentum
for i in range(n_iterations):
    y_pred = X @ beta_momentum
    error = y_pred - y
    cost = (1 / len(y)) * np.sum(error**2)
    cost_history_momentum.append(cost)
    gradient = (2 / len(y)) * X.T @ error
    velocity = gamma * velocity + learning_rate * gradient
    beta_momentum -= velocity

plt.plot(range(n_iterations), cost_history, label="Plain GD")
plt.plot(range(n_iterations), cost_history_momentum, label="GD with Momentum")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost Function Convergence")
plt.legend()
plt.show()

print("Final beta coefficients (GD with Momentum):")
print(beta_momentum)

print("Final beta coefficients:")
print(beta)
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# Newton's Method implementation
beta_newton = np.random.randn(X.shape[1], 1)
n_iterations_newton = 10
cost_history_newton = []

# Precompute the Hessian matrix and its inverse
H = (2 / len(y)) * X.T @ X
H_inv = np.linalg.inv(H)

### Using the eigen value of Hessian matrix to calculate the learning rate
### Instead of using the inverse of the Hessian matrix
# H_eign_values, H_Eign_vec = np.linalg.eig(H)
# max_eign = np.max(H_eign_values)
# inver_max_eign = 1 / max_eign

for i in range(n_iterations_newton):
    y_pred = X @ beta_newton
    error = y_pred - y
    cost = (1 / len(y)) * np.sum(error**2)
    cost_history_newton.append(cost)
    gradient = (2 / len(y)) * X.T @ error
    beta_newton -= H_inv @ gradient
    # beta_newton -= inver_max_eign * gradient

plt.plot(range(n_iterations_newton), cost_history_newton)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost Function Convergence (Newtons Method")
plt.show()

print("Final beta coefficients (Newton's Method):")
print(beta_newton)
