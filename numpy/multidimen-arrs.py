import numpy as np

# ==========================================================
# 1️⃣ CREATING MULTIDIMENSIONAL ARRAYS
# ==========================================================

# 1D Array (Vector)
a1 = np.array([1, 2, 3])
print("1D Array:\n", a1, "\nShape:", a1.shape)

# 2D Array (Matrix)
a2 = np.array([[1, 2, 3], [4, 5, 6]])
print("\n2D Array:\n", a2, "\nShape:", a2.shape)

# 3D Array (Tensor)
a3 = np.array([
    [[1, 2, 3], [4, 5, 6]],
    [[7, 8, 9], [10, 11, 12]]
])
print("\n3D Array:\n", a3, "\nShape:", a3.shape)

# Create arrays filled with zeros or ones
zeros = np.zeros((2, 3))
ones = np.ones((3, 2))
print("\nZeros Array:\n", zeros)
print("\nOnes Array:\n", ones)

# Random array
random_array = np.random.rand(2, 3)
print("\nRandom Array:\n", random_array)

# ==========================================================
# 2️⃣ INDEXING & SLICING
# ==========================================================

# Simple indexing
print("\nSingle element (2D):", a2[0, 1])  # 2

# Multi-level indexing (3D)
print("Single element (3D):", a3[1, 0, 2])  # 9

# Slicing 2D
print("\nFirst row of a2:", a2[0, :])
print("Second column of a2:", a2[:, 1])

# Slicing 3D
print("\nFirst block of a3:\n", a3[0])
print("All layers, first row:\n", a3[:, 0, :])

# ==========================================================
# 3️⃣ ITERATING THROUGH MULTIDIM ARRAYS
# ==========================================================

print("\nIterating through 2D array:")
for row in a2:
    print(row)

print("\nIterating through 3D array (layer by layer):")
for layer in a3:
    print(layer)

# Flat iteration (go through every element)
print("\nFlat iteration over all elements:")
for x in np.nditer(a3):
    print(x, end=" ")

# ==========================================================
# 4️⃣ RESHAPING ARRAYS
# ==========================================================

a = np.arange(12)
print("\n\nOriginal 1D array:\n", a)

a_reshaped = a.reshape(3, 4)
print("\nReshaped to 3x4:\n", a_reshaped)

a_reshaped_3d = a.reshape(2, 3, 2)
print("\nReshaped to 3D (2,3,2):\n", a_reshaped_3d)

# Flatten back to 1D
a_flat = a_reshaped.flatten()
print("\nFlattened array:\n", a_flat)

# ==========================================================
# 5️⃣ AXES AND AGGREGATIONS
# ==========================================================

b = np.array([[1, 2, 3], [4, 5, 6]])
print("\nArray b:\n", b)

# Sum along rows (axis=1)
print("Sum along rows:", np.sum(b, axis=1))

# Sum along columns (axis=0)
print("Sum along columns:", np.sum(b, axis=0))

# Mean, min, max
print("Mean:", np.mean(b))
print("Min:", np.min(b))
print("Max:", np.max(b))

# ==========================================================
# 6️⃣ BROADCASTING (Automatic Expansion)
# ==========================================================

x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([10, 20, 30])  # Broadcasted to match shape of x
print("\nBroadcasting example:\n", x + y)

# ==========================================================
# 7️⃣ TRANSPOSE & SWAP AXES
# ==========================================================

print("\nOriginal 2D array:\n", a2)
print("Transposed:\n", a2.T)

# For 3D arrays
a3_T = np.transpose(a3, (1, 0, 2))
print("\n3D Transpose (swap first two axes):\n", a3_T)

# ==========================================================
# 8️⃣ COPYING VS VIEWING ARRAYS
# ==========================================================

arr = np.arange(6).reshape(2, 3)
view = arr.view()   # Shares memory
copy = arr.copy()   # Independent copy

arr[0, 0] = 99
print("\nOriginal array:\n", arr)
print("View (affected):\n", view)
print("Copy (not affected):\n", copy)

# ==========================================================
# 9️⃣ DIMENSION INFO
# ==========================================================

print("\nDimensions of a3:", a3.ndim)
print("Shape of a3:", a3.shape)
print("Size (total elements):", a3.size)
print("Data type:", a3.dtype)

# ==========================================================
# 🔟 Example
# ==========================================================

array = np.array([
    [[1,2,3],[1,2,3]],
    [[4,5,6],[4,5,6]],
    [[7,8,9],[7,8,9]]
])

code = array[0,0,0] + array[1,0,0] + array[2,0,0] + array[2,1,0]
print(code)
