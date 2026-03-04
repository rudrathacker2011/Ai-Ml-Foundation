# ========== COMPLETE NUMPY REFERENCE ==========
import numpy as np

print("="*50)
print("1. CREATING ARRAYS")
print("="*50)

# 1D array
arr_1d = np.array([1, 2, 3, 4, 5, 6, 7, 8])
print("1D array:", arr_1d)

# 2D array
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
print("2D array:\n", arr_2d)
print("\n")

# ========== 2. SHAPE AND DIMENSION ==========
print("="*50)
print("2. SHAPE AND DIMENSION")
print("="*50)

print("arr_1d.shape:", arr_1d.shape)      # (8,) → 1D, 8 elements
print("arr_2d.shape:", arr_2d.shape)      # (3, 3) → 3 rows, 3 columns
print("arr_1d.ndim:", arr_1d.ndim)        # 1 → 1 dimension
print("arr_2d.ndim:", arr_2d.ndim)        # 2 → 2 dimensions
print("\n")

# ========== 3. AXIS OPERATIONS ==========
print("="*50)
print("3. AXIS OPERATIONS")
print("="*50)

print("Sum down rows (axis=0):", arr_2d.sum(axis=0))     # [12, 15, 18]
print("Sum across columns (axis=1):", arr_2d.sum(axis=1)) # [6, 15, 24]
print("Mean along axis=0:", arr_2d.mean(axis=0))
print("Mean along axis=1:", arr_2d.mean(axis=1))
print("Max along axis=0:", arr_2d.max(axis=0))
print("Min along axis=1:", arr_2d.min(axis=1))
print("\n")

# ========== 4. INDEXING & SLICING ==========
print("="*50)
print("4. INDEXING & SLICING (1D)")
print("="*50)

print("arr_1d[0] (first element):", arr_1d[0])           # 1
print("arr_1d[-1] (last element):", arr_1d[-1])          # 8
print("arr_1d[-2] (second last):", arr_1d[-2])           # 7
print("arr_1d[2:7] (slice 3rd to 7th):", arr_1d[2:7])   # [3 4 5 6 7]
print("arr_1d[:4] (first 4):", arr_1d[:4])               # [1 2 3 4]
print("arr_1d[5:] (from 6th onwards):", arr_1d[5:])      # [6 7 8]
print("arr_1d[::2] (every 2nd element):", arr_1d[::2])   # [1 3 5 7]
print("\n")

print("="*50)
print("5. INDEXING & SLICING (2D)")
print("="*50)

print("arr_2d[0, 1] (row 0, col 1):", arr_2d[0, 1])      # 2
print("arr_2d[:, 1] (all rows, col 1):", arr_2d[:, 1])   # [2 5 8]
print("arr_2d[1, :] (row 1, all cols):", arr_2d[1, :])   # [4 5 6]
print("arr_2d[0:2, 1:3] (submatrix):\n", arr_2d[0:2, 1:3])
print("arr_2d[-1, -1] (last element):", arr_2d[-1, -1])  # 9
print("\n")

# ========== 6. VECTORIZATION (No loops needed) ==========
print("="*50)
print("6. VECTORIZATION")
print("="*50)

result_multiply = arr_1d * 2
print("arr_1d * 2:", result_multiply)

result_power = arr_1d ** 2
print("arr_1d ** 2:", result_power)

result_add = arr_1d + 10
print("arr_1d + 10:", result_add)

result_div = arr_1d / 2
print("arr_1d / 2:", result_div)

# Element-wise operations on 2D
result_2d = arr_2d * 3
print("arr_2d * 3:\n", result_2d)
print("\n")

# ========== 7. AGGREGATIONS ==========
print("="*50)
print("7. AGGREGATIONS")
print("="*50)

print("arr_1d.sum():", arr_1d.sum())           # 36
print("arr_1d.mean():", arr_1d.mean())         # 4.5
print("arr_1d.min():", arr_1d.min())           # 1
print("arr_1d.max():", arr_1d.max())           # 8
print("arr_1d.std():", arr_1d.std())           # Standard deviation
print("arr_1d.cumsum():", arr_1d.cumsum())     # Cumulative sum [1, 3, 6, 10...]
print("\n")

print("arr_2d.sum():", arr_2d.sum())           # Total sum
print("arr_2d.mean():", arr_2d.mean())         # Overall mean
print("arr_2d.min():", arr_2d.min())           # Overall min
print("arr_2d.max():", arr_2d.max())           # Overall max
print("\n")

# ========== 8. RESHAPING ==========
print("="*50)
print("8. RESHAPING")
print("="*50)

reshaped_2x4 = arr_1d.reshape(2, 4)
print("arr_1d.reshape(2, 4):\n", reshaped_2x4)

reshaped_4x2 = arr_1d.reshape(4, 2)
print("arr_1d.reshape(4, 2):\n", reshaped_4x2)

# Auto-calculate dimension with -1
reshaped_auto = arr_1d.reshape(4, -1)
print("arr_1d.reshape(4, -1):\n", reshaped_auto)  # Automatically (4, 2)

# Column vector (important for ML)
column_vector = arr_1d.reshape(-1, 1)
print("arr_1d.reshape(-1, 1) - Column vector:\n", column_vector)
print("Shape:", column_vector.shape)  # (8, 1)

# Row vector
row_vector = arr_1d.reshape(1, -1)
print("arr_1d.reshape(1, -1) - Row vector:\n", row_vector)
print("Shape:", row_vector.shape)  # (1, 8)

# Flatten back to 1D
flattened = arr_2d.flatten()
print("arr_2d.flatten():", flattened)
print("\n")

# ========== 9. BROADCASTING ==========
print("="*50)
print("9. BROADCASTING")
print("="*50)

add_this = np.array([1, 2, 3])
result_broadcast = arr_2d + add_this
print("arr_2d + [1, 2, 3]:\n", result_broadcast)
# [1, 2, 3] is broadcast to each row

# Broadcasting with column vector
col_add = np.array([[10], [20], [30]])
result_col_broadcast = arr_2d + col_add
print("arr_2d + [[10], [20], [30]]:\n", result_col_broadcast)
# Added 10 to row 0, 20 to row 1, 30 to row 2

# Scalar broadcasting
result_scalar = arr_2d + 100
print("arr_2d + 100:\n", result_scalar)
print("\n")

# ========== 10. BOOLEAN INDEXING ==========
print("="*50)
print("10. BOOLEAN INDEXING")
print("="*50)

mask = arr_1d > 4
print("arr_1d > 4:", mask)  # [False False False False True True True True]
print("arr_1d[arr_1d > 4]:", arr_1d[arr_1d > 4])  # [5 6 7 8]

# Multiple conditions
result_and = arr_1d[(arr_1d > 2) & (arr_1d < 7)]
print("arr_1d > 2 AND < 7:", result_and)  # [3 4 5 6]

result_or = arr_1d[(arr_1d < 3) | (arr_1d > 6)]
print("arr_1d < 3 OR > 6:", result_or)  # [1 2 7 8]
print("\n")

# ========== 11. CREATING SPECIAL ARRAYS ==========
print("="*50)
print("11. CREATING SPECIAL ARRAYS")
print("="*50)

zeros = np.zeros((3, 3))
print("np.zeros((3, 3)):\n", zeros)

ones = np.ones((2, 4))
print("np.ones((2, 4)):\n", ones)

identity = np.eye(3)
print("np.eye(3) - Identity matrix:\n", identity)

arange = np.arange(0, 10, 2)  # Start, stop, step
print("np.arange(0, 10, 2):", arange)  # [0 2 4 6 8]

linspace = np.linspace(0, 1, 5)  # 5 evenly spaced values between 0 and 1
print("np.linspace(0, 1, 5):", linspace)

random_array = np.random.rand(2, 3)  # Random values between 0 and 1
print("np.random.rand(2, 3):\n", random_array)

random_int = np.random.randint(0, 100, size=(3, 3))  # Random integers
print("np.random.randint(0, 100, size=(3, 3)):\n", random_int)
print("\n")

# ========== 12. ARRAY OPERATIONS ==========
print("="*50)
print("12. ARRAY OPERATIONS")
print("="*50)

arr_a = np.array([1, 2, 3])
arr_b = np.array([4, 5, 6])

print("arr_a + arr_b:", arr_a + arr_b)  # Element-wise addition
print("arr_a - arr_b:", arr_a - arr_b)  # Element-wise subtraction
print("arr_a * arr_b:", arr_a * arr_b)  # Element-wise multiplication
print("arr_a / arr_b:", arr_a / arr_b)  # Element-wise division

# Dot product
print("np.dot(arr_a, arr_b):", np.dot(arr_a, arr_b))  # 1*4 + 2*5 + 3*6 = 32

# Matrix multiplication (for 2D)
mat_a = np.array([[1, 2], [3, 4]])
mat_b = np.array([[5, 6], [7, 8]])
print("Matrix multiplication:\n", np.dot(mat_a, mat_b))
# OR using @ operator
print("mat_a @ mat_b:\n", mat_a @ mat_b)
print("\n")

# ========== 13. STACKING & CONCATENATION ==========
print("="*50)
print("13. STACKING & CONCATENATION")
print("="*50)

arr_x = np.array([1, 2, 3])
arr_y = np.array([4, 5, 6])

# Vertical stack (stack rows)
vstacked = np.vstack([arr_x, arr_y])
print("np.vstack([arr_x, arr_y]):\n", vstacked)

# Horizontal stack (stack columns)
hstacked = np.hstack([arr_x, arr_y])
print("np.hstack([arr_x, arr_y]):", hstacked)

# Concatenate
concatenated = np.concatenate([arr_x, arr_y])
print("np.concatenate([arr_x, arr_y]):", concatenated)
print("\n")

# ========== 14. TRANSPOSING ==========
print("="*50)
print("14. TRANSPOSING")
print("="*50)

print("Original arr_2d:\n", arr_2d)
print("arr_2d.T (transposed):\n", arr_2d.T)
print("\n")

# ========== 15. PRACTICAL ML EXAMPLE ==========
print("="*50)
print("15. COMPLETE ML WORKFLOW EXAMPLE")
print("="*50)

# Create sample dataset
X = np.array([[1, 2],
              [3, 4],
              [5, 6],
              [7, 8]])
y = np.array([0, 0, 1, 1])

print("Features (X):\n", X)
print("Shape:", X.shape)  # (4, 2) → 4 samples, 2 features
print("\nTarget (y):", y)
print("Shape:", y.shape)  # (4,)

# Normalize features: (X - mean) / std
mean = X.mean(axis=0)
std = X.std(axis=0)
X_normalized = (X - mean) / std  # Broadcasting in action
print("\nNormalized X:\n", X_normalized)

print("\nReady for: model.fit(X_normalized, y)")
print("\n")

# ========== SUMMARY TABLE ==========
print("="*50)
print("QUICK REFERENCE TABLE")
print("="*50)
print("""
Operation               | Code
------------------------|----------------------------------
Create array            | np.array([1, 2, 3])
Shape                   | arr.shape
Dimensions              | arr.ndim
Holding rows for sum    | arr.sum(axis=0)
Holding cols for sum    | arr.sum(axis=1)
Indexing                | arr[0], arr[0, 1]
Slicing                 | arr[2:7], arr[:, 1]
Multiply all by 2       | arr * 2
Reshape                 | arr.reshape(2, 4)
Auto-reshape            | arr.reshape(-1, 1)
Broadcasting            | arr + [1, 2, 3]
Boolean indexing        | arr[arr > 5]
Create zeros            | np.zeros((3, 3))
Create ones             | np.ones((2, 4))
Random array            | np.random.rand(3, 3)
Transpose               | arr.T
Flatten                 | arr.flatten()
Stack vertically        | np.vstack([arr1, arr2])
Stack horizontally      | np.hstack([arr1, arr2])
""")