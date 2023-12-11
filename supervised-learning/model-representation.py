# Goals: Learn to implement the model f -sub {w,b} for linear regression with one variable.

import numpy as np
import matplotlib.pyplot as plt

# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]  # could also use len(x_train)
print(f"Number of training examples is: {m}")

# Plot the data points
plt.scatter(x_train, y_train, marker="x", c="r")
plt.title("Housing Prices")
plt.ylabel("Price (1000s of dollars)")
plt.xlabel("Size (1000 sqft)")
plt.show()

# Testing out w and b
w = 200
b = 100


def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
        x (ndarray (m,)): Data, m examples
        w,b (scalar)    : model parameters
    Returns
        f_wb (ndarray (m,)): model prediction
    """
    m = len(x)
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb


tmp_f_wb = compute_model_output(x_train, w, b)
plt.plot(x_train, tmp_f_wb, c="b", label="Our Prediction")
plt.scatter(x_train, y_train, marker="x", c="r", label="Actual Values")
plt.title("Housing Prices")
plt.ylabel("Price")
plt.xlabel("Sqft")
plt.legend()
plt.show()

# Prediction
x_i = 1.2
cost_1200sqft = w * x_i + b
print(f"${cost_1200sqft:.0f} thousand dollars")

# Recap
# Linear regression builds a model which establishes a relationship between features and targets
# In this example, house size was the feature and the target was house price
# For simple linear regression, the model has two parameters w and b whose values are 'fit' using training data
# Once a model's parameters have been determined, the model can be used to make predictions on novel data
