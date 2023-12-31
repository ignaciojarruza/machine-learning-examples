# Goals: Implement and explore the cost function for linear regression with one variable
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480, 430, 630, 730])


# J(w,b)
def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray (m,)): Data, m examples
        y (ndarray (m,)): target values
        w,b (scalar)    : model parameters

    Returns
        total_cost (float): The cost of using w,b as the parameters for
        linear regression to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0]

    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum += cost
    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost


print(compute_cost(x_train, y_train, 200.0, 100.0))
# For interactive visualization examples using Jupyter Notebooks and ipympl, follow Cost Function Visualization Lab from
# Andrew Ng's Machine Learning Course on Regression/Classification.
