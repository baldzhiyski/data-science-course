# ------------------------------------------------------------

import numpy as np
np.set_printoptions(precision=4, suppress=True)  # cleaner prints

print("\n================ 0) RECOMMENDED API & REPRODUCIBILITY ================\n")
# default_rng(seed) creates a *Generator* object with its own state.
# If you pass the SAME seed, you get the SAME sequence of random values.
rng = np.random.default_rng(seed=42)  # <- change the seed to get a different sequence
print("We created rng = np.random.default_rng(seed=42)")
print("Example call rng.random(3):", rng.random(3))
print("If you re-run the program with the same seed, you'll see the same numbers here.\n")

# You can also pick a specific BitGenerator (advanced). PCG64 is a good default:
rng_pcg = np.random.Generator(np.random.PCG64(12345))
print("Alternative generator using PCG64 (also reproducible with seed=12345):", rng_pcg.random(3), "\n")


print("\n================ 1) UNIFORM RANDOM FLOATS ================\n")
# rng.random(size) -> floats in the half-open interval [0.0, 1.0)
# IMPORTANT: 1.0 is NOT included.
print("rng.random() single float in [0,1):", rng.random())
print("rng.random(5) 1D array of 5 floats:", rng.random(5))
print("rng.random((2,3)) 2x3 matrix of floats:\n", rng.random((2,3)))

# rng.uniform(low, high, size) -> floats in [low, high)
# IMPORTANT: 'high' is NOT included (half-open interval), just like Python's range().
print("rng.uniform(-1.5, 2.5, (2,4)) -> shape (2,4), range [-1.5, 2.5):\n", rng.uniform(-1.5, 2.5, (2,4)))


print("\n================ 2) RANDOM INTEGERS (DISCRETE) ================\n")
# rng.integers(low, high, size) -> integers in [low, high)  (high is EXCLUDED)
# Example: 1..9 (10 is NOT included)
ints_2x3 = rng.integers(1, 10, size=(2,3))
print("rng.integers(1, 10, size=(2,3)) ->\n", ints_2x3)
print("Check: min >= 1 and max <= 9:", ints_2x3.min(), ints_2x3.max())

# If you want to INCLUDE the high endpoint, set endpoint=True
ints_inclusive = rng.integers(1, 10, size=5, endpoint=True)
print("rng.integers(1, 10, size=5, endpoint=True) -> includes 10 possible:", ints_inclusive, "\n")


print("\n================ 3) CORE CONTINUOUS DISTRIBUTIONS ================\n")
# NORMAL / GAUSSIAN:
# rng.standard_normal(size): mean≈0, std≈1
# rng.normal(loc, scale, size): mean=loc, std=scale
std_norm = rng.standard_normal(5)
norm_2x3 = rng.normal(loc=10, scale=2, size=(2,3))
print("Standard Normal ~ N(0,1):", std_norm)
print("Normal with mean=10, std=2 (2x3):\n", norm_2x3)
print("Empirical mean of above matrix (roughly 10):", norm_2x3.mean(), "  std (roughly 2):", norm_2x3.std())

# LOGNORMAL:
# If Y ~ Normal(mean, sigma), then X = exp(Y) is Lognormal(mean, sigma)
logn = rng.lognormal(mean=0.0, sigma=0.5, size=5)
print("\nLognormal (mean=0,sigma=0.5):", logn, "  (always positive)")

# EXPONENTIAL:
# rng.exponential(scale) with mean = scale  (scale = 1/lambda in rate notation)
expv = rng.exponential(scale=2.0, size=5)
print("Exponential (scale=2.0, mean≈2.0):", expv)

# GAMMA:
# rng.gamma(shape=k, scale=theta);  mean = k*theta, var = k*theta^2
gamm = rng.gamma(shape=2.0, scale=2.0, size=5)
print("Gamma(k=2.0, theta=2.0) mean≈4.0:", gamm)

# BETA:
# rng.beta(a, b) -> values in (0,1). Shape controlled by a and b.
bet = rng.beta(a=2.0, b=5.0, size=5)
print("Beta(a=2, b=5) in (0,1):", bet, "\n")


print("\n================ 4) CORE DISCRETE DISTRIBUTIONS ================\n")
# BINOMIAL:
# n trials, probability p of success; returns number of successes per draw.
binom = rng.binomial(n=10, p=0.3, size=6)
print("Binomial(n=10, p=0.3) -> counts of successes:", binom)

# POISSON:
# Count of events happening in a fixed interval with average rate 'lam'.
pois = rng.poisson(lam=4.0, size=6)
print("Poisson(lam=4.0) -> counts around 4:", pois)

# GEOMETRIC:
# Number of trials until first success (1-indexed) with probability p.
geom = rng.geometric(p=0.25, size=6)
print("Geometric(p=0.25) -> trial index of first success:", geom)

# HYPERGEOMETRIC:
# Drawing without replacement from a finite population:
# ngood = #good items, nbad = #bad items, nsample = draws per trial.
hyper = rng.hypergeometric(ngood=7, nbad=12, nsample=5, size=6)
print("Hypergeometric(ngood=7, nbad=12, nsample=5):", hyper, "\n")


print("\n================ 5) RANDOM CHOICE (CATEGORICAL) ================\n")
# rng.choice(options, size, replace=True/False, p=probabilities)
# - replace=True: can pick the same item multiple times
# - replace=False: picks are unique (like sampling without replacement)
animals = np.array(["cat", "dog", "bird"])
print("Pick 5 animals (with replacement):", rng.choice(animals, size=5))

print("Pick 2 animals (without replacement):", rng.choice(animals, size=2, replace=False))

# Weighted choice: probabilities must sum to 1
probs = np.array([0.6, 0.3, 0.1])  # cat more likely
weighted = rng.choice(animals, size=10, p=probs)
print("Weighted choice (p=[0.6,0.3,0.1]):", weighted, "\n")


print("\n================ 6) SHUFFLING vs PERMUTATION ================\n")
# permutation(x) -> returns a shuffled COPY (original unchanged)
# shuffle(x)     -> shuffles IN PLACE (original is modified)
arr = np.arange(10)
perm = rng.permutation(arr)  # copy
print("Original arr:", arr)
print("rng.permutation(arr) (copy):", perm)

rng.shuffle(arr)  # in-place
print("After rng.shuffle(arr) (in-place):", arr)

# Permute rows/columns of a 2D array with index permutations
X = np.arange(12).reshape(4, 3)
row_idx = rng.permutation(X.shape[0])
col_idx = rng.permutation(X.shape[1])
print("\nX:\n", X)
print("Row order:", row_idx, "  Col order:", col_idx)
print("Row-permuted:\n", X[row_idx, :])
print("Col-permuted:\n", X[:, col_idx], "\n")


print("\n================ 7) TRAIN / VAL / TEST SPLIT (TYPICAL ML) ================\n")
# One common pattern: permute indices 0..n-1, then slice ranges for splits.
n = 20
idx = rng.permutation(n)
train_end = int(0.7 * n)   # 70% train
val_end   = int(0.85 * n)  # 15% val, 15% test
train_idx, val_idx, test_idx = idx[:train_end], idx[train_end:val_end], idx[val_end:]
print("Index split sizes -> train:", train_idx.size, "val:", val_idx.size, "test:", test_idx.size)
print("train_idx head:", train_idx[:5])


print("\n================ 8) RANDOM MASKS (e.g., DROPOUT) ================\n")
# A random boolean mask with probability p of True
p = 0.2  # 20% chance to be True
mask = rng.random(10) < p
print("Random mask (~20% True):", mask)
# Example use: zero-out masked positions
v = np.arange(10, dtype=float)
masked_v = np.where(mask, 0.0, v)
print("Original v:", v)
print("Zeroed where mask is True:", masked_v, "\n")


print("\n================ 9) BOOTSTRAPPING (RESAMPLE WITH REPLACEMENT) ================\n")
# Bootstrap = sample the same array with replacement, same length
data = np.array([3.2, 1.1, 5.6, 2.4, 9.0, 4.5])
boot = rng.choice(data, size=len(data), replace=True)
print("Original data:", data)
print("Bootstrap sample (same length, with replacement):", boot)
print("Notice duplicates may appear; some original items may be missing.\n")


print("\n================ 10) PROBABILITY VECTORS & COUNTS ================\n")
# DIRICHLET:
# Generates probability vectors that sum to 1 (useful for topic mixtures, etc.)
alpha = np.array([2.0, 1.0, 3.0])    # concentration params (higher -> more even)
theta = rng.dirichlet(alpha, size=4) # 4 samples of 3-class probability vectors
print("Dirichlet samples (each row sums to 1):\n", theta)
print("Row sums (should be 1.0 each):", theta.sum(axis=1))

# MULTINOMIAL:
# Given probabilities p (sum to 1), draw counts from n trials.
counts = rng.multinomial(n=10, pvals=[0.5, 0.3, 0.2], size=5)
print("Multinomial counts (5 draws, 10 trials each):\n", counts)
print("Row sums (each should equal 10):", counts.sum(axis=1), "\n")


print("\n================ 11) MULTIVARIATE NORMAL (CORRELATED VARS) ================\n")
# You can sample 2D/3D normals with a mean vector and covariance matrix.
mean = np.array([0.0, 1.0])
cov  = np.array([[ 1.0, 0.8],
                 [ 0.8, 2.0]])   # positive correlation 0.8
samples = rng.multivariate_normal(mean, cov, size=5)
print("Samples (shape 5x2):\n", samples)
print("Sample mean (approx -> [0,1]):", samples.mean(axis=0))
print("Sample covariance (approx to cov):\n", np.cov(samples, rowvar=False), "\n")


print("\n================ 12) PARAMETER TIPS (MEAN vs SCALE vs RATE) ================\n")
# Many distributions use "scale" rather than "mean".
# Examples:
# - Exponential(scale=1/lambda). Mean = scale.
# - Gamma(shape=k, scale=theta). Mean = k*theta, Var = k*theta^2.
# - Lognormal(mean, sigma) are parameters of the *underlying* Normal before exp().
print("Tip: Always check what the parameters mean for each distribution (docs!).\n")


print("\n================ 13) PARALLEL-SAFE SEEDING (ADVANCED) ================\n")
# If you need multiple independent random streams (e.g., workers/threads),
# use a SeedSequence to *spawn* child seeds. Each child produces independent streams.
ss = np.random.SeedSequence(2025)
child_seqs = ss.spawn(3)  # make 3 independent child sequences
workers = [np.random.Generator(np.random.PCG64(s)) for s in child_seqs]
print("Three independent RNG streams (each row is a different stream):")
for i, w in enumerate(workers):
    print(f"  worker {i} draws:", w.integers(0, 100, size=5))
print("(Re-running with the same top SeedSequence will reproduce these.)\n")


print("\n================ 14) SHAPES, DTYPES, BROADCASTING NOTES ================\n")
# Most Generator methods accept a 'size' argument to control the shape.
# Some accept dtype explicitly (e.g., rng.random), others infer (e.g., rng.integers).
print("Float32 randoms 2x3:\n", rng.random((2,3), dtype=np.float32))
print("Small ints int16:\n", rng.integers(0, 100, size=6, dtype=np.int16))
# Broadcasting: many NumPy operations can combine these random arrays with others element-wise.


print("\n================ 15) LEGACY API NOTE (STILL WORKS) ================\n")
# Legacy global functions (np.random.uniform, np.random.randn, etc.) still work:
legacy_uniform = np.random.uniform()   # float in [0,1)
print("Legacy np.random.uniform() example:", legacy_uniform)
# BUT: they use a single *global* RNG state. Prefer the Generator API for clarity and safety.


print("\n================ 16) COMMON PITFALLS & QUICK CHECKS ================\n")
# 1) Off-by-one errors for integers:
ints = rng.integers(1, 5, size=20)  # 1,2,3,4 (5 is EXCLUDED)
print("Check integer bounds [1,5): min:", ints.min(), "max:", ints.max())

# 2) Forgetting shapes:
print("Shape examples -> rng.random(5).shape:", rng.random(5).shape, " | rng.random((2,3)).shape:", rng.random((2,3)).shape)

# 3) Expectation checks (does sample average look reasonable?)
draw = rng.normal(loc=100, scale=15, size=10000)
print("Sanity check for normal(100,15): sample mean≈", round(draw.mean(), 2), " sample std≈", round(draw.std(), 2))

print("\n✅ Finished. You now have a guided tour of NumPy random with lots of comments.\n")
