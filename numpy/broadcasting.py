# ==========================================================
# broadcasting_examples.py
# ==========================================================
# This file demonstrates ALL major uses of NumPy broadcasting
# ==========================================================

import numpy as np

print("\n================ 1) BASIC BROADCASTING (SCALAR) ================\n")

# A scalar is automatically broadcast to every element of an array
A = np.array([[1, 2, 3],
              [4, 5, 6]])

print("Original array A:\n", A)
print("Shape of A:", A.shape)

# Add a scalar (broadcast to all elements)
print("\nA + 10:\n", A + 10)

# Multiply by scalar
print("\nA * 2:\n", A * 2)

# Broadcasting works the same for all element-wise operators (+, -, *, /, **, etc.)
print("\nA ** 2:\n", A ** 2)


print("\n================ 2) ROW-WISE BROADCASTING ================\n")

# A 1D array can broadcast across rows if its length matches the number of columns
b = np.array([10, 20, 30])  # shape (3,)
print("Array b:", b, "Shape:", b.shape)

print("\nA + b (b broadcast across rows):\n", A + b)
print("Result shape:", (A + b).shape)


print("\n================ 3) COLUMN-WISE BROADCASTING ================\n")

# Add a column vector (n,1) across columns
c = np.array([[100],
              [200]])  # shape (2,1)
print("Array c:\n", c, "Shape:", c.shape)

print("\nA + c (c broadcast across columns):\n", A + c)
print("Result shape:", (A + c).shape)


print("\n================ 4) ROW * COLUMN COMBINATION ================\n")

# Multiply row vector (1,3) by column vector (3,1)
array1 = np.array([[1, 2, 3]])  # shape (1,3)
array2 = np.array([[1],
                   [2],
                   [3]])        # shape (3,1)

print("array1 shape:", array1.shape)
print("array2 shape:", array2.shape)
print("\narray1 * array2 (broadcast to 3x3):\n", array1 * array2)
# Explanation:
# array1 -> expanded to (3,3)
# array2 -> expanded to (3,3)
# Result  -> (3,3)


print("\n================ 5) DIFFERENT DIMENSIONS (EXPANDING WITH None) ================\n")

# Use np.newaxis (or None) to add dimensions manually
x = np.array([1, 2, 3])   # shape (3,)
y = np.array([10, 20, 30, 40])  # shape (4,)

print("x shape:", x.shape)
print("y shape:", y.shape)

# Make x a column vector (3,1) and y a row vector (1,4)
print("\nx[:, None] + y[None, :]:\n", x[:, None] + y[None, :])
print("Shape of result:", (x[:, None] + y[None, :]).shape)

# Outer product via broadcasting
print("\nOuter product (x[:,None] * y[None,:]):\n", x[:, None] * y[None, :])


print("\n================ 6) 3D BROADCASTING ================\n")

A3 = np.ones((2, 3, 4))  # shape (2,3,4)
B3 = np.arange(4)        # shape (4,)
print("A3 shape:", A3.shape)
print("B3 shape:", B3.shape)

print("\nA3 * B3:\n", A3 * B3)
print("Result shape:", (A3 * B3).shape)
# (2,3,4) * (4,) -> (2,3,4)


print("\n================ 7) BROADCASTING IN MACHINE LEARNING STYLE OPS ================\n")

# Example: normalizing columns
X = np.array([[1., 2., 3.],
              [4., 5., 6.],
              [7., 8., 9.]])  # shape (3,3)
print("X:\n", X)

# Column means (shape (3,))
col_mean = X.mean(axis=0)
print("\nColumn means:", col_mean)

X_centered = X - col_mean  # broadcast across rows
print("\nX_centered = X - col_mean:\n", X_centered)

# Add a bias per feature (row vector)
bias = np.array([0.5, -1.0, 2.0])  # shape (3,)
print("\nX + bias:\n", X + bias)


print("\n================ 8) BROADCASTING IN IMAGE-LIKE OPERATIONS ================\n")

# Imagine a grayscale image (height x width)
image = np.ones((2, 3))
brightness = 50  # scalar broadcast
contrast = np.array([[1], [1.5]])  # (2,1)

print("Original image:\n", image)
print("\nimage * contrast + brightness:\n", image * contrast + brightness)


print("\n================ 9) PAIRWISE DIFFERENCES OR DISTANCES ================\n")

points1 = np.array([[0, 0],
                    [1, 1],
                    [2, 2]])  # shape (3,2)
points2 = np.array([[0, 1],
                    [1, 0]])  # shape (2,2)

# Compute all pairwise differences
diff = points1[:, None, :] - points2[None, :, :]  # (3,2,2)
print("Pairwise differences shape:", diff.shape)
print("\nPairwise differences:\n", diff)

# Compute pairwise squared distances
dist_sq = np.sum(diff**2, axis=2)
print("\nSquared distances (3x2):\n", dist_sq)


print("\n================ 10) BROADCASTING WITH FUNCTIONS (UFUNCS) ================\n")

arr = np.array([[0, np.pi/2, np.pi]])
print("arr:\n", arr)

# np.sin and np.cos automatically broadcast
print("\nnp.sin(arr):\n", np.sin(arr))
print("\nnp.sin(arr) + np.cos(arr):\n", np.sin(arr) + np.cos(arr))


print("\n================ 11) BROADCASTING WITH DIFFERENT DTYPE ARRAYS ================\n")

ints = np.array([[1, 2, 3]], dtype=np.int32)
floats = np.array([[0.1], [0.2], [0.3]], dtype=np.float64)

print("ints shape:", ints.shape, "dtype:", ints.dtype)
print("floats shape:", floats.shape, "dtype:", floats.dtype)
print("\nints + floats (automatic upcast & broadcast):\n", ints + floats)


print("\n================ 12) MEMORY EFFICIENCY NOTE ================\n")

big = np.ones((1000, 1000))
small = np.arange(1000)
print("Result shape of big + small:", (big + small).shape)
print("=> small is not copied 1000 times, it's broadcast virtually in memory!")


print("\n✅ All broadcasting examples completed successfully!\n")
