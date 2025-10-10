# ==========================================================
# numpy_aggregates_examples.py
# ==========================================================
# A tour of NumPy aggregate (reduction) functions with
# shapes, axis behavior, NaN handling, dtype control,
# keepdims, where/out, weighted average, percentiles, etc.
# ==========================================================

import numpy as np

np.set_printoptions(precision=3, suppress=True)

print("\n================ 0) Setup arrays ================\n")
A = np.arange(1, 13).reshape(3, 4)      # 3x4: 1..12
print("A:\n", A, "\nshape:", A.shape)

B = np.array([[1, 2, np.nan, 4],
              [5, np.nan, 7, 8],
              [9, 10, 11, 12]], dtype=float)
print("\nB (with NaNs):\n", B)

C = np.array([[True, False, True],
              [False, False, True]])
print("\nC (bools):\n", C)

# ------------------------------------------------------------
# 1) SUM / PROD
# ------------------------------------------------------------
print("\n================ 1) Sum / Prod ================\n")
print("np.sum(A):", np.sum(A))
print("A.sum(axis=0)  # column sums:", A.sum(axis=0))
print("A.sum(axis=1)  # row sums   :", A.sum(axis=1))
print("np.prod(A, axis=1) # row products:", np.prod(A, axis=1))

# dtype control (avoid integer overflow with smaller ints)
small = np.array([1, 2, 3], dtype=np.int8)
print("\nsmall:", small, "dtype:", small.dtype)
print("sum small (default int8 may overflow):", small.sum())
print("sum small with dtype=np.int32:", small.sum(dtype=np.int32))

# initial for empty reductions (e.g., maximum of empty slice)
empty = np.array([], dtype=int)
print("\nnp.sum(empty, initial=0):", np.sum(empty, initial=0))

# ------------------------------------------------------------
# 2) MEAN / MEDIAN / MIN / MAX / PTP
# ------------------------------------------------------------
print("\n================ 2) Mean/Median/Min/Max/PTP ================\n")
print("A.mean():", A.mean())
print("A.mean(axis=0):", A.mean(axis=0))
print("np.median(A):", np.median(A))
print("np.median(A, axis=1):", np.median(A, axis=1))
print("A.min(), A.max():", A.min(), A.max())
print("A.min(axis=0):", A.min(axis=0))
print("A.max(axis=1):", A.max(axis=1))
print("np.ptp(A)  # max - min over entire array:", np.ptp(A))
print("np.ptp(A, axis=0):", np.ptp(A, axis=0))

# ------------------------------------------------------------
# 3) ARGMIN / ARGMAX (+ unravel_index)
# ------------------------------------------------------------
print("\n================ 3) Argmin/Argmax ================\n")
idx_max = A.argmax()   # flat index
idx_min = A.argmin()
print("A.argmax() ->", idx_max, "| A.argmin() ->", idx_min)

# Convert flat index to 2D index
print("np.unravel_index(A.argmax(), A.shape) ->", np.unravel_index(idx_max, A.shape))
print("np.unravel_index(A.argmin(), A.shape) ->", np.unravel_index(idx_min, A.shape))

# Along an axis
print("A.argmax(axis=0) ->", A.argmax(axis=0))
print("A.argmin(axis=1) ->", A.argmin(axis=1))

# ------------------------------------------------------------
# 4) STD / VAR (population vs sample)
# ------------------------------------------------------------
print("\n================ 4) Std / Var ================\n")
print("A.std(), A.var():", A.std(), A.var())
print("A.std(axis=0):", A.std(axis=0))
print("A.var(axis=1):", A.var(axis=1))

# ddof = 1 for sample std/var (unbiased estimator)
print("Sample std (ddof=1) over axis=0:", A.std(axis=0, ddof=1))
print("Sample var (ddof=1) over axis=1:", A.var(axis=1, ddof=1))

# ------------------------------------------------------------
# 5) ANY / ALL / COUNT_NONZERO (boolean reductions)
# ------------------------------------------------------------
print("\n================ 5) Any / All / Count_nonzero ================\n")
print("C.any()  ->", C.any())
print("C.all()  ->", C.all())
print("C.any(axis=0) ->", C.any(axis=0))
print("C.all(axis=1) ->", C.all(axis=1))
print("np.count_nonzero(C):", np.count_nonzero(C))
print("np.count_nonzero(A % 2 == 0)  # number of even entries:", np.count_nonzero(A % 2 == 0))

# ------------------------------------------------------------
# 6) NaN-safe variants: nanmean, nanstd, nanmin, etc.
# ------------------------------------------------------------
print("\n================ 6) NaN-safe reductions ================\n")
print("np.mean(B)      # NaN propagates ->", np.mean(B))
print("np.nanmean(B)   # ignores NaNs   ->", np.nanmean(B))
print("np.nanstd(B), np.nanvar(B):", np.nanstd(B), np.nanvar(B))
print("np.nanmin(B, axis=0):", np.nanmin(B, axis=0))
print("np.nanmax(B, axis=1):", np.nanmax(B, axis=1))
print("np.nansum(B, axis=0):", np.nansum(B, axis=0))

# Replace NaNs for later operations
print("np.nan_to_num(B, nan=-1):\n", np.nan_to_num(B, nan=-1))

# ------------------------------------------------------------
# 7) PERCENTILE / QUANTILE
# ------------------------------------------------------------
print("\n================ 7) Percentile / Quantile ================\n")
print("np.percentile(A, 50)  # median:", np.percentile(A, 50))
print("np.percentile(A, [0, 25, 50, 75, 100]):", np.percentile(A, [0, 25, 50, 75, 100]))
print("np.percentile(A, 90, axis=0):", np.percentile(A, 90, axis=0))

# quantile is the same idea (0..1 instead of 0..100)
print("np.quantile(A, [0.25, 0.5, 0.75], axis=1):\n", np.quantile(A, [0.25, 0.5, 0.75], axis=1))

# With NaNs: use nanpercentile/nanquantile
print("np.nanpercentile(B, 50):", np.nanpercentile(B, 50))

# ------------------------------------------------------------
# 8) AVERAGE (weighted mean)
# ------------------------------------------------------------
print("\n================ 8) Weighted average ================\n")
X = np.array([[1., 2., 3.],
              [4., 5., 6.],
              [7., 8., 9.]])
w = np.array([0.2, 0.3, 0.5])   # weights per column
print("X:\n", X)
print("weights w (per column):", w)

print("np.average(X, axis=0, weights=w):", np.average(X, axis=0, weights=w))
# If weights along rows (per row), change axis accordingly:
wr = np.array([1.0, 2.0, 1.0])
print("np.average(X, axis=1, weights=wr):", np.average(X, axis=1, weights=wr))

# ------------------------------------------------------------
# 9) AXIS tips: None vs int vs tuple, keepdims
# ------------------------------------------------------------
print("\n================ 9) Axis & keepdims ================\n")
print("A.sum()            ->", A.sum(), "shape: scalar")
print("A.sum(axis=0)      ->", A.sum(axis=0), "shape:", A.sum(axis=0).shape)
print("A.sum(axis=1)      ->", A.sum(axis=1), "shape:", A.sum(axis=1).shape)
print("A.sum(axis=(0,1))  ->", A.sum(axis=(0,1)), "(sum over rows and cols)")

print("\nkeepdims=True keeps reduced axes as size-1:")
print("A.sum(axis=1, keepdims=True):\n", A.sum(axis=1, keepdims=True),
      "shape:", A.sum(axis=1, keepdims=True).shape)

# ------------------------------------------------------------
# 10) WHERE and OUT parameters
# ------------------------------------------------------------
print("\n================ 10) where= and out= parameters ================\n")
# Sum only values > 6 (others treated as 0)
mask = A > 6
print("mask (A>6):\n", mask)
print("np.sum(A, where=mask):", np.sum(A, where=mask))

# Use out= to store result in a preallocated array
target = np.zeros(4, dtype=np.int64)
np.sum(A, axis=0, out=target)
print("out= target (column sums placed here):", target)

# ------------------------------------------------------------
# 11) CUMULATIVE vs AGGREGATE
# ------------------------------------------------------------
print("\n================ 11) Cumulative vs Aggregate ================\n")
row = np.array([1, 2, 3, 4])
print("row:", row)
print("row.sum()          ->", row.sum())
print("row.cumsum()       ->", row.cumsum())
print("row.prod()         ->", row.prod())
print("row.cumprod()      ->", row.cumprod())

# Along axis for 2D
print("A.cumsum(axis=0):\n", A.cumsum(axis=0))
print("A.cumsum(axis=1):\n", A.cumsum(axis=1))

# ------------------------------------------------------------
# 12) HISTOGRAMS & BINCOUNTS (useful aggregations)
# ------------------------------------------------------------
print("\n================ 12) Histogram / Bincount ================\n")
vals = np.array([0, 1, 1, 2, 3, 3, 3, 4])
print("vals:", vals)
print("np.bincount(vals):", np.bincount(vals))  # counts each integer label

d = np.array([1.0, 0.5, 2.0, 1.5, 1.0, 1.0, 0.5, 2.0])
print("Weighted bincount:", np.bincount(vals, weights=d))

# Histogram for continuous ranges
data = np.array([0.1, 0.4, 0.8, 1.2, 1.9, 2.1, 2.5])
hist, edges = np.histogram(data, bins=4, range=(0.0, 2.5))
print("np.histogram counts:", hist)
print("np.histogram edges :", edges)

# ------------------------------------------------------------
# 13) GROUP-LIKE SUM with reduceat (advanced)
# ------------------------------------------------------------
print("\n================ 13) Group-like sum with reduceat ================\n")
# Suppose we want sums over segments [0:3), [3:6), [6:8)
arr = np.arange(1, 9)  # 1..8
idx = np.array([0, 3, 6])
seg_sum = np.add.reduceat(arr, idx)
print("arr:", arr)
print("segment starts:", idx)
print("segment sums via reduceat:", seg_sum, "# last segment ends at end")

# ------------------------------------------------------------
# 14) PERFORMANCE NOTE: methods vs np.func
# ------------------------------------------------------------
print("\n================ 14) Methods vs functions ================\n")
# A.sum() and np.sum(A) are equivalent for most purposes.
# np.sum has extra args like 'where'/'initial' (newer NumPy),
# but both call fast C-level reductions.
print("A.sum() == np.sum(A):", A.sum() == np.sum(A))

print("\n✅ All aggregate examples completed successfully!\n")
