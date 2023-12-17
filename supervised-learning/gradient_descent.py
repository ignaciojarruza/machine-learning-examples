# Goals: Automate the process of optimizing w and b using gradient descent
import math, copy
import numpy as np
import matplotlib.pyplot as plt
from cost_function import compute_cost
from resources.lab_utils_uni import (
    plt_house_x,
    plt_contour_wgrad,
    plt_divergence,
    plt_gradients,
)

# Small data set
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])


def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression
    Args:
        x (ndarray (m,)): Data, m examples
        y (ndarray (m,)): target values
        w, b (scalar)   : model parameters
    Returns:
        dj_dw (scalar)  : The gradient of the cost w.r.t the parameters w
        df_db (scalar)  : The gradient of the cost w.r.t the parameter b
    """

    # Number of training examples
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


plt_gradients(x_train, y_train, compute_cost, compute_gradient)
plt.show()
