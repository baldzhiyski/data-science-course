# boolean_indexing_and_where_examples_verbose.py
# ------------------------------------------------------------
# Goal: Learn boolean indexing, masks, np.where/select, bins,
#       counts, NaN handling, and common patterns on arrays.
#       Every step is commented in plain language.
# ------------------------------------------------------------

import numpy as np

# Make prints easier to read
np.set_printoptions(suppress=True)

# ============================================================
# 0) SETUP: a small 2x8 array of ages
#    - 2 rows (e.g., two groups)
#    - 8 columns (e.g., people per group)
# ============================================================
ages = np.array([
    [21, 17, 19, 20, 16, 30, 18, 65],
    [39, 22, 15, 99, 18, 19, 20, 21]
])
print("Ages (2 rows x 8 columns):\n", ages)
print("Shape of ages:", ages.shape)  # (2, 8)


# ============================================================
# 1) BASIC BOOLEAN MASKS (True/False arrays)
#    A "mask" is an array of True/False with the same shape.
#    When you index with a mask, you get the values where the
#    mask is True (flattened).
# ============================================================

# Mask for "age <= 18" (teenagers)
mask_teen = (ages <= 18)            # shape (2,8) of True/False
print("\nMask for teenagers (<= 18):\n", mask_teen)

# Use the mask to select values (returns 1D array of matched values)
teenagers = ages[mask_teen]
print("Teenagers values (flattened):", teenagers)

# Mask for "18 <= age < 65" (adults)
mask_adult = (ages >= 18) & (ages < 65)
print("\nMask for adults (>=18 and <65):\n", mask_adult)

adults = ages[mask_adult]
print("Adults values (flattened):", adults)

# Mask for "age >= 65" (seniors)
mask_senior = (ages >= 65)
print("\nMask for seniors (>=65):\n", mask_senior)

seniors = ages[mask_senior]
print("Seniors values (flattened):", seniors)

# Mask for "even ages" (divisible by 2)
mask_even = (ages % 2 == 0)
print("\nMask for even ages:\n", mask_even)

evens = ages[mask_even]
print("Even ages (flattened):", evens)


# ============================================================
# 2) np.where(condition, value_if_true, value_if_false)
#    - This creates a new array with the same shape as 'ages'.
#    - Where condition is True:   take value_if_true
#    - Where condition is False:  take value_if_false
# ============================================================

# Keep age if >= 18, otherwise put -1 (just a placeholder)
adults_or_minus1 = np.where(ages >= 18, ages, -1)
print("\nnp.where example (>=18 keep age else -1):\n", adults_or_minus1)
# Note: original 'ages' is unchanged (np.where returns a new array)


# ============================================================
# 3) NESTED WHERE and np.select for multiple rules
#    We want to label each element as:
#    - 0: child/teen (<18)
#    - 1: adult (18..64)
#    - 2: senior (>=65)
# ============================================================

# Using nested np.where (works, but can look noisy with many rules)
labels_where = np.where(ages >= 65, 2,                   # condition1 -> 2
                 np.where((ages >= 18) & (ages < 65), 1, # condition2 -> 1
                          0))                            # else -> 0
print("\nLabels via nested np.where (0 teen, 1 adult, 2 senior):\n", labels_where)

# Using np.select (cleaner when you have many conditions)
conditions = [
    (ages >= 65),               # condition A
    (ages >= 18) & (ages < 65)  # condition B
]
choices = [2, 1]                # values for A and B; default below
labels_select = np.select(conditions, choices, default=0)
print("Labels via np.select (same meaning):\n", labels_select)


# ============================================================
# 4) IN-PLACE EDITS USING MASKS
#    - Copy the array if you want to preserve the original.
#    - Use mask to change only some positions.
# ============================================================

ages_edit = ages.copy()
# Example rule: set ages < 18 to 0 (mask controls *where* to write)
ages_edit[ages_edit < 18] = 0
print("\nIn-place edit (set under-18 to 0):\n", ages_edit)

# Another example: add +1 year to all adults (>=18 and <65)
ages_birthday = ages.copy()
ages_birthday[mask_adult] += 1
print("Add 1 to all adult ages (in-place on a copy):\n", ages_birthday)


# ============================================================
# 5) MEMBERSHIP TESTS WITH np.isin
#    - Keep/Drop values based on a set of allowed/blocked values.
# ============================================================

allowed = np.array([16, 18, 21, 30, 65, 99])  # a small set of "interesting" ages
mask_allowed = np.isin(ages, allowed)         # True where ages ∈ allowed
print("\nMask of ages that are in", allowed, ":\n", mask_allowed)

keep_allowed = ages[mask_allowed]  # just the matching values (flattened)
print("Values that are in the allowed set:", keep_allowed)

# Exclude some values (negate the mask with ~)
exclude_21_99 = ages[~np.isin(ages, [21, 99])]
print("Values excluding 21 and 99:", exclude_21_99)


# ============================================================
# 6) FINDING INDICES OF MATCHES
#    - np.argwhere returns a list of [row, col] for True positions.
#    - np.nonzero returns (rows, cols) arrays; same meanings.
# ============================================================

print("\nIndices (row,col) of seniors (>=65):")
idx_seniors = np.argwhere(ages >= 65)
print(idx_seniors)  # e.g., [[0,7], [1,3]]

print("Indices (rows, cols) of even ages (from nonzero on mask):")
even_rows, even_cols = np.nonzero(mask_even)
print("rows:", even_rows, "cols:", even_cols)
print("Values at those positions:", ages[even_rows, even_cols])  # fancy indexing


# ============================================================
# 7) AXIS-AWARE LOGIC: any() and all()
#    - any(axis=1): for each row, is there at least one True?
#    - all(axis=1): for each row, are all entries True?
# ============================================================

# For each row: is there any teenager?
has_teen_per_row = (ages <= 18).any(axis=1)
print("\nPer-row: has at least one teenager? ->", has_teen_per_row)

# For each row: are all people adults? (18..64)
all_adults_per_row = ((ages >= 18) & (ages < 65)).all(axis=1)
print("Per-row: are all people adults (18..64)? ->", all_adults_per_row)


# ============================================================
# 8) NaN HANDLING (missing values)
#    - NaN means “missing”.
#    - Use np.isnan to build masks.
#    - Use nan-safe functions (nanmean, nanmin, ...)
# ============================================================

ages_nan = ages.astype(float)  # must be float to hold NaN
ages_nan[0, 2] = np.nan        # put a NaN at row 0, col 2 (just for demo)
print("\nAges with a NaN injected at [0,2]:\n", ages_nan)

nan_mask = np.isnan(ages_nan)
print("NaN mask (True = is NaN):\n", nan_mask)

# nanmean ignores NaN values when computing the mean
print("Row-wise mean ignoring NaNs:", np.nanmean(ages_nan, axis=1))

# Replace NaNs with a chosen value (e.g., -1) using where
filled = np.where(np.isnan(ages_nan), -1, ages_nan)
print("NaNs replaced with -1:\n", filled)


# ============================================================
# 9) BUCKETING / BINNING WITH np.digitize
#    - Turn continuous values into categories by ranges.
#    Example bins: [0,13,18,65,inf) -> child, teen, adult, senior
#    NOTE: we assume ages are >= 0 here.
# ============================================================

bins = np.array([0, 13, 18, 65, np.inf])  # 4 intervals
# digitize returns an integer 1..len(bins) indicating which bin edge we passed
bucket_ids = np.digitize(ages, bins, right=False)
print("\nBins:", bins)
print("digitize (bin indices with same shape as ages):\n", bucket_ids)

# Map bin indices to human-readable labels
bin_labels = np.array(["<0?", "child", "teen", "adult", "senior"])
print("Labels (same shape as ages):\n", bin_labels[bucket_ids])


# ============================================================
# 10) COUNTING VALUES
#     a) np.unique(..., return_counts=True): counts unique values
#     b) np.bincount: fast counts for non-negative integers
#     c) np.histogram: counts in numeric ranges (bins)
# ============================================================

# Flatten to 1D for counting (easier to read)
flat = ages.ravel()

# a) Unique values + counts
uvals, counts = np.unique(flat, return_counts=True)
print("\nUnique ages:", uvals)
print("Counts      :", counts)

# b) Bincount: index = age, value = count (only works for non-negative ints)
#    minlength ensures the array is long enough to include the highest age index
bc = np.bincount(flat, minlength=flat.max()+1)
nonzero_idx = np.nonzero(bc)[0]
print("\nbincount (only showing ages that appear):")
print("ages  :", nonzero_idx)
print("count:", bc[nonzero_idx])

# c) Histogram by decades (0-10, 10-20, 20-30, ..., up to 120)
hist, edges = np.histogram(flat, bins=[0,10,20,30,40,50,60,70,80,100,120])
print("\nHistogram counts per decade bin:", hist)
print("Bin edges:", edges)


# ============================================================
# 11) SAFE TRANSFORMS USING MASKS
#     - Example 1: clamp all ages into the range [18, 80]
#     - Example 2: change only odd numbers (demo)
# ============================================================

# Clamp: values < 18 become 18; values > 80 become 80
clamped = np.clip(ages, 18, 80)
print("\nClamped ages to [18, 80]:\n", clamped)

# Modify only odd numbers: subtract 1 to make them even (just to show masked edit)
odd_mask = (ages % 2 == 1)
adjusted = ages.copy()
adjusted[odd_mask] = adjusted[odd_mask] - 1
print("Change odd -> previous even (masked edit):\n", adjusted)


# ============================================================
# 12) SORT ONLY SOME POSITIONS (using a mask)
#     - Example: sort only adult values inside their positions,
#       leave non-adult positions unchanged.
# ============================================================

sorted_demo = ages.copy()
adult_values_only = sorted_demo[mask_adult]     # grab the adult values
sorted_adults = np.sort(adult_values_only)      # sort those values
sorted_demo[mask_adult] = sorted_adults         # put them back at adult positions
print("\nSort only adult positions (others stay as-is):\n", sorted_demo)


# ============================================================
# 13) COMBINING MULTIPLE RULES INTO ONE MASK
#     - Build masks step-by-step with |= (OR) and &= (AND).
# ============================================================

complex_mask = np.zeros_like(ages, dtype=bool)  # start with all False
complex_mask |= (ages < 18)   # add under-18
complex_mask |= (ages >= 90)  # add very old (>=90)
print("\nComplex mask (under-18 OR >=90):\n", complex_mask)
print("Values matching the complex mask:\n", ages[complex_mask])


print("\n✅ Finished: boolean masks, where/select, indices, bins, counts, NaN, edits.\n")
