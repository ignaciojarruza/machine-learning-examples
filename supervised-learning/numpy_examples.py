# A brief introduction to some of the scientific computing used in ML.
# In particular the NumPy scientific computing package and its uses with python.

import numpy as np
import time

# Numpy routines which allocate memory and fill arrays with value
a = np.zeros(4)
print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.zeros((4,))
print(f"np.zeros((4,)) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.random_sample(4)
print(
    f"np.random.random_sample(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}"
)
