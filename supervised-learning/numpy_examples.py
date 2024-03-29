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

# NumPy routines which allocate memory andf fill arrays with value but do not accept shape as input
a = np.arange(4.0)
print(f"np.arange(4.):     a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.rand(4)
print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

# Indexing
# Vector indexing operations on 1-D vectors
a = np.arange(10)
print(a)

# access an element
print(f"a[2].shape: {a[2].shape} a[2] = {a[2]}, Accessing an element returns a scalar")

# Access the last element, negative indexes count from the end
print(f"a[-1] = {a[-1]}")

# Indexes must be within the range of the vector or they will produce an error
try:
    c = a[10]
except Exception as e:
    print("The error message you'll see is:")
    print(e)

# Slicing
# vector slicing operations
a = np.arange(10)
print(f"a       = {a}")
# access 5 consecutive elements (start:stop:step) -> up to but not including end
c = a[2:7:1]
print("a[2:7:1] = ", c)
# access 3 elements separated by two
c = a[2:7:2]
print("a[2:7:2] = ", c)
# access all elemends index 3 and above
c = a[3:]
print("a[3:] =", c)
# access all elements below index 3
c = a[:3]
print("a[:3]    = ", c)
# access all elements
c = a[:]
print("a[:]     = ", c)

# Single Vector Operations
a = np.array([1, 2, 3, 4])
print(f"a             : {a}")
b = -a
print(f"b = -a        : {b}")
b = np.sum(a)
print(f"b = np.sum(a) : {b}")
b = np.mean(a)
print(f"b = np.mean(a): {b}")
b = a**2
print(f"b = a**2      : {b}")

# Vector Vector element-wise operations
a = np.array([1, 2, 3, 4])
b = np.array([-1, -2, 3, 4])
print(f"Binary operators work element wise: {a + b}")
c = np.array([1, 2])
try:
    d = a + c
except Exception as e:
    print("Error message for mismatched vectors:")
    print(e)

# Scalar Vector Operations
a = np.array([1, 2, 3, 4])
b = 5 * a
print(f"b = 5 * a : {b}")


# Dot Product Comparison
# Manual dot product that does not use vectorization
def my_dot(a, b):
    """
    Compute the dot product of two vectors

    Args:
        a (ndarray(n,)):    input vector
        b (ndarray(n,)):    input vector

    Returns:
        x (scalar):
    """
    x = 0
    for i in range(a.shape[0]):
        x = x + a[i] * b[i]
    return x


a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
print(f"my_dot(a, b) = {my_dot(a, b)}")

# Using np.dot
c = np.dot(a, b)
print(f"NumPy 1-D np.dot(a, b) = {c}, np.dot(a, b).shape = {c.shape} ")
c = np.dot(b, a)
print(f"NumPy 1-D np.dot(b, a) = {c}, np.dot(a, b).shape = {c.shape} ")

# Time comparison: Manual vs. vectorization dot product
np.random.seed(1)
a = np.random.rand(10000000)  # very large arrays
b = np.random.rand(10000000)

tic = time.time()  # capture start time
c = np.dot(a, b)
toc = time.time()  # capture end time

print(f"np.dot(a, b) =  {c:.4f}")
print(f"Vectorized version duration: {1000*(toc-tic):.4f} ms ")

tic = time.time()  # capture start time
c = my_dot(a, b)
toc = time.time()  # capture end time

print(f"my_dot(a, b) =  {c:.4f}")
print(f"loop version duration: {1000*(toc-tic):.4f} ms ")

del a
del b  # remove these big arrays from memory
