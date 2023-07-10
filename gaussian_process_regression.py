import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Generate some sample data
np.random.seed(0)
X_train = np.linspace(-5, 5, 20)
y_train = np.sin(X_train) + np.random.randn(len(X_train)) * 0.1

# Define the Gaussian Process model
kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1)

# Fit the model to the training data
gp.fit(X_train[:, np.newaxis], y_train)

# Generate test data
X_test = np.linspace(-7, 7, 100)

# Predict using the trained model
y_pred, sigma = gp.predict(X_test[:, np.newaxis], return_std=True)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, c='red', label='Training Data')
plt.plot(X_test, np.sin(X_test), c='green', label='True Function')
plt.plot(X_test, y_pred, c='blue', label='Predicted Function')
plt.fill_between(X_test, y_pred - 2 * sigma, y_pred + 2 * sigma, color='gray', alpha=0.3, label='Uncertainty')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Gaussian Process Regression')
plt.legend()
plt.show()