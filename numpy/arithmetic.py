import numpy as np

print("\n================ 0) Setup ================\n")
array = np.array([1, 2, 3], dtype=np.int64)
print("array:", array, "dtype:", array.dtype)

# ------------------------------------------------------------
# 1) Scalar arithmetic: applies element-wise to every entry
# ------------------------------------------------------------
print("\n================ 1) Scalar arithmetic ================\n")
print("array ** 4 ->", array ** 4)      # power
print("array + 2  ->", array + 2)       # add
print("array - 2  ->", array - 2)       # subtract
print("array * 2  ->", array * 2)       # multiply
print("array / 2  ->", array / 2)       # true division -> float dtype
print("array // 2 ->", array // 2)      # floor division -> int dtype
print("array % 2  ->", array % 2)       # modulo
print("(-array)   ->", -array)          # unary minus

# Equivalent ufunc calls (same results, more options like 'out=')
print("\nUsing ufuncs (np.add/np.multiply/etc.):")
print("np.add(array, 2)   ->", np.add(array, 2))
print("np.power(array, 4) ->", np.power(array, 4))

# ------------------------------------------------------------
# 2) Vectorized math functions (ufuncs): fast, element-wise
# ------------------------------------------------------------
print("\n================ 2) Vectorized math functions (ufuncs) ================\n")
arr = np.array([0.0, 1.0, 2.0, 3.0])
print("arr:", arr)

print("np.sqrt(arr)   ->", np.sqrt(arr))
print("np.sin(arr)    ->", np.sin(arr))
print("np.cos(arr)    ->", np.cos(arr))
print("np.tan(arr)    ->", np.tan(arr))
print("np.arctan(arr) ->", np.arctan(arr))
print("np.exp(arr)    ->", np.exp(arr))
print("np.log(arr+1)  ->", np.log(arr + 1))  # avoid log(0)
print("np.round(np.pi * arr**2, 2) ->", np.round(np.pi * arr**2, 2))

# degree ↔ radian helpers
deg = np.array([0, 30, 90, 180])
rad = np.deg2rad(deg)
print("\nDegrees:", deg, "-> radians:", rad)
print("np.sin(rad) ->", np.sin(rad))

# ------------------------------------------------------------
# 3) Element-wise array arithmetic (same shape or broadcastable)
# ------------------------------------------------------------
print("\n================ 3) Element-wise arithmetic ================\n")
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])
print("array1:", array1)
print("array2:", array2)

print("array1 + array2 ->", array1 + array2)
print("array1 - array2 ->", array1 - array2)
print("array1 * array2 ->", array1 * array2)  # element-wise multiply
print("array1 / array2 ->", array1 / array2)

# ------------------------------------------------------------
# 4) Broadcasting: align shapes by prepending 1s where possible
# ------------------------------------------------------------
print("\n================ 4) Broadcasting ================\n")
A = np.arange(12).reshape(3, 4)
b = np.array([10, 20, 30, 40])     # shape (4,)
c = np.array([[100], [200], [300]])# shape (3,1)

print("A:\n", A)
print("b:", b, "shape:", b.shape)
print("c:\n", c, "shape:", c.shape)

print("\nA + b (row-wise add):\n", A + b)   # broadcasts over rows
print("\nA + c (col-wise add):\n", A + c)   # broadcasts over columns
print("\n(A + b) + c:\n", (A + b) + c)

# Pitfall: incompatible shapes raise a ValueError
try:
    _ = A + np.array([1, 2, 3])  # shape (3,) won't align with (3,4)
except ValueError as e:
    print("\nBroadcasting error example:", e)

# ------------------------------------------------------------
# 5) Comparison operators & boolean masks (filter/assign)
# ------------------------------------------------------------
print("\n================ 5) Comparisons & boolean masks ================\n")
scores = np.array([91, 55, 100, 73, 82, 64])
print("scores:", scores)

print("scores == 100 ->", scores == 100)
print("scores < 60   ->", scores < 60)

# Update values where condition is true
scores_copy = scores.copy()
scores_copy[scores_copy < 60] = 0
print("scores after setting <60 to 0 ->", scores_copy)

# Combining conditions: use & (and), | (or), ~ (not) with parentheses
mask = (scores >= 80) & (scores <= 90)
print("80 <= scores <= 90 mask ->", mask)
print("scores[mask] ->", scores[mask])

# np.where for conditional selection
curved = np.where(scores < 60, scores + 10, scores)  # add 10 bonus to failing grades
print("curved (where <60 add 10) ->", curved)

# ------------------------------------------------------------
# 6) NaN-safe math (use nan-versions to ignore NaNs)
# ------------------------------------------------------------
print("\n================ 6) NaN-safe math ================\n")
x = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
print("x:", x)
print("np.mean(x)      ->", np.mean(x))        # propagates NaN -> NaN
print("np.nanmean(x)   ->", np.nanmean(x))     # ignores NaNs
print("np.nansum(x)    ->", np.nansum(x))
print("np.nan_to_num(x, nan=-1) ->", np.nan_to_num(x, nan=-1.0))

# ------------------------------------------------------------
# 7) Dtypes, upcasting, overflow, and precision
# ------------------------------------------------------------
print("\n================ 7) Dtypes, upcasting, overflow ================\n")
i8 = np.array([120], dtype=np.int8)
print("int8 start:", i8, i8.dtype)
print("int8 + 10   ->", (i8 + 10), (i8 + 10).dtype)   # still int8
# print("int8 + 200  ->", (i8 + 200), (i8 + 200).dtype) # wraparound/overflow!

# Upcasting: result dtype accommodates both operands
ints = np.array([1, 2, 3], dtype=np.int32)
floats = np.array([0.5, 1.5, 2.5], dtype=np.float64)
print("\nints dtype:", ints.dtype, "floats dtype:", floats.dtype)
print("ints + floats ->", ints + floats, "dtype:", (ints + floats).dtype)

# Be explicit to avoid surprises
print("np.add(ints, floats, dtype=np.float64) ->", np.add(ints, floats, dtype=np.float64))

# ------------------------------------------------------------
# 8) In-place vs out-of-place operations
# ------------------------------------------------------------
print("\n================ 8) In-place vs out-of-place ================\n")
arr = np.array([1, 2, 3], dtype=np.int64)
print("arr start:", arr, arr.dtype)

arr2 = arr + 5           # out-of-place (new array)
print("arr2 = arr + 5  ->", arr2, "(arr unchanged:", arr, ")")

arr += 5                 # in-place (modifies arr)
print("arr += 5        ->", arr, "(same object modified)")

# In-place with ufuncs and 'out' parameter
target = np.empty_like(arr)
np.multiply(arr, 2, out=target)    # target = arr * 2
print("out parameter (target = arr*2):", target)

# ------------------------------------------------------------
# 9) Element-wise vs matrix multiplication
# ------------------------------------------------------------
print("\n================ 9) Element-wise vs matrix multiplication ================\n")
M = np.array([[1, 2], [3, 4]])
N = np.array([[10, 20], [30, 40]])
print("M:\n", M)
print("N:\n", N)

print("\nElement-wise M * N:\n", M * N)    # Hadamard product
print("Matrix multiply M @ N:\n", M @ N)  # matrix product
print("np.dot(M, N) (same as M @ N for 2D):\n", np.dot(M, N))

# ------------------------------------------------------------
# 10) Reductions (sum/mean/etc.) and accumulate
# ------------------------------------------------------------
print("\n================ 10) Reductions & accumulate ================\n")
X = np.arange(1, 7).reshape(2, 3)
print("X:\n", X)
print("sum all      ->", X.sum())
print("sum axis=0   ->", X.sum(axis=0))  # column-wise
print("sum axis=1   ->", X.sum(axis=1))  # row-wise
print("mean axis=1  ->", X.mean(axis=1))
print("min, max     ->", X.min(), X.max())

# ufunc.reduce and accumulate
print("\nnp.add.reduce(X, axis=1) ->", np.add.reduce(X, axis=1))       # row sums
print("np.multiply.accumulate([1,2,3,4]) ->", np.multiply.accumulate([1,2,3,4]))

# ------------------------------------------------------------
# 11) Clipping, rounding, and other handy element-wise ops
# ------------------------------------------------------------
print("\n================ 11) Clipping & rounding ================\n")
y = np.array([-3.7, -0.2, 0.1, 1.5, 9.9])
print("y:", y)
print("np.clip(y, 0, 2)   ->", np.clip(y, 0, 2))
print("np.floor(y)        ->", np.floor(y))
print("np.ceil(y)         ->", np.ceil(y))
print("np.round(y, 1)     ->", np.round(y, 1))
print("np.abs(y)          ->", np.abs(y))

# ------------------------------------------------------------
# 12) Safer conditional transforms with boolean masks
# ------------------------------------------------------------
print("\n================ 12) Conditional transforms ================\n")
z = np.array([12, 58, 61, 73, 89, 40])
print("z:", z)

# Curve only sub-60 and cap everything at 90
z2 = z.copy()
z2[z2 < 60] += 10
z2 = np.clip(z2, None, 90)
print("curved & capped ->", z2)

# OR use where-chaining
curved2 = np.where(z < 60, z + 10, z)
capped2 = np.minimum(curved2, 90)
print("curved2 ->", curved2, " | capped2 ->", capped2)

# ------------------------------------------------------------
# 13) Exp
# ------------------------------------------------------------
print("\n================ 13) Given snippet recap ================\n")
array1 = np.array([1,2,3])
array2 = np.array([4,5,6])
print("array1 + array2 ->", array1 + array2)
print("array1 - array2 ->", array1 - array2)
print("array1 * array2 ->", array1 * array2)

scores = np.array([91,55,100,73,82,64])
print("scores == 100 ->", scores == 100)
print("scores < 60   ->", scores < 60)
scores[scores < 60] = 0
print("scores after <60 -> 0:", scores)

print("\n✅ All sections executed.")
