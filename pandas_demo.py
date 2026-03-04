import numpy as np
import pandas as pd

# ========== 1. Creating DataFrame ==========
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 55],
    'Score': [90, 78, 99]
}
df_raw = pd.DataFrame(data)
print("Original DataFrame:")
print(df_raw)
print("\n")

# ========== 2. Loading CSV ==========
df = pd.read_csv(r"C:\Users\thack\OneDrive\Documents\Python\students.csv")
print("Loaded CSV - First 5 rows:")
print(df.head())
print("\n")

print("Last 3 rows:")
print(df.tail(3))
print("\n")

print("Random 2 samples:")
print(df.sample(2))
print("\n")

# ========== 3. Understanding data — info, describe ==========
print("DataFrame Info:")
print(df.info())
print("\n")

print("Statistical Summary:")
print(df.describe())  # Uses NumPy mean, std, min, max internally
print("\n")

# ========== 4. Column selection ==========
# Single column (returns Series)
ages = df_raw['Age']
print("Single column (Series):")
print(ages)
print("Type:", type(ages))
print("As NumPy array:", ages.values)
print("\n")

# Multiple columns (returns DataFrame)
subset = df_raw[['Name', 'Score']]
print("Multiple columns:")
print(subset)
print("\n")

# ========== 5. Adding new column (Boolean indexing) ==========
df_raw['Pass'] = df_raw['Score'] >= 91
print("After adding 'Pass' column:")
print(df_raw)
print("\n")

# ========== 6. Dropping column ==========
df_raw = df_raw.drop(columns=['Pass'])
print("After dropping 'Pass' column:")
print(df_raw)
print("\n")

# ========== 7. Filtering rows (Boolean indexing) ==========
age_filtered = df_raw[df_raw['Age'] > 26]
print("Rows where Age > 26:")
print(age_filtered)
print("\n")

# Multiple conditions
multi_filter = df_raw[(df_raw['Age'] > 26) & (df_raw['Score'] > 80)]
print("Age > 26 AND Score > 80:")
print(multi_filter)
print("\n")

# OR condition
or_filter = df_raw[(df_raw['Age'] < 30) | (df_raw['Score'] > 95)]
print("Age < 30 OR Score > 95:")
print(or_filter)
print("\n")

# ========== 8. iloc and loc indexing ==========
print(".iloc[0, 1] - Position based (row 0, col 1):")
print(df_raw.iloc[0, 1])  # 25
print("\n")

print(".loc[0, 'Age'] - Label based:")
print(df_raw.loc[0, 'Age'])  # 25
print("\n")

print("First 2 rows, columns 'Name' and 'Score':")
print(df_raw.loc[0:1, ['Name', 'Score']])
print("\n")

# ========== 9. Converting to NumPy ==========
X = df_raw[['Age', 'Score']].values
print("Converted to NumPy array:")
print(X)
print("Type:", type(X))
print("Shape:", X.shape)
print("\n")

# ========== 10. Aggregations (NumPy under the hood) ==========
print("Column-wise mean (axis=0):")
print(df_raw[['Age', 'Score']].mean())
print("\n")

print("Sum of all scores:")
print(df_raw['Score'].sum())
print("\n")

print("Min and Max age:")
print("Min:", df_raw['Age'].min())
print("Max:", df_raw['Age'].max())
print("\n")

# ========== 11. Handling missing data ==========
# Create sample with missing values
data_missing = {
    'Name': ['Alice', 'Bob', None, 'David'],
    'Age': [25, np.nan, 35, 40],
    'Score': [90, 78, np.nan, 85]
}
df_missing = pd.DataFrame(data_missing)

print("DataFrame with missing values:")
print(df_missing)
print("\n")

print("Check missing values:")
print(df_missing.isna())
print("\n")

print("Count missing per column:")
print(df_missing.isna().sum())
print("\n")

# Drop rows with any missing value
df_dropped = df_missing.dropna()
print("After dropna():")
print(df_dropped)
print("\n")

# Fill missing values
df_filled = df_missing.fillna(0)
print("After fillna(0):")
print(df_filled)
print("\n")

# Fill Age with mean
df_missing['Age'] = df_missing['Age'].fillna(df_missing['Age'].mean())
print("After filling Age with mean:")
print(df_missing)
print("\n")

# ========== 12. Shape and axis (NumPy connection) ==========
print("DataFrame shape:")
print(df_raw.shape)  # (3, 3) — 3 rows, 3 columns
print("\n")

# Sum along axis
numeric_df = df_raw[['Age', 'Score']]
print("Sum along axis=0 (column-wise):")
print(numeric_df.sum(axis=0))
print("\n")

print("Sum along axis=1 (row-wise):")
print(numeric_df.sum(axis=1))
print("\n")

# ========== 13. Reshaping (via NumPy) ==========
ages_array = df_raw['Age'].values
print("Original age array:", ages_array)
print("Shape:", ages_array.shape)  # (3,)
print("\n")

reshaped = ages_array.reshape(-1, 1)  # Column vector
print("Reshaped to column vector:")
print(reshaped)
print("New shape:", reshaped.shape)  # (3, 1)
print("\n")

# ========== 14. Vectorization (Pandas on top of NumPy) ==========
# Add bonus to all scores (vectorized operation)
df_raw['Score_with_bonus'] = df_raw['Score'] + 10
print("After adding 10 to all scores:")
print(df_raw)
print("\n")

# Drop the bonus column
df_raw = df_raw.drop(columns=['Score_with_bonus'])

# ========== 15. Broadcasting example ==========
# Normalize scores (subtract mean, divide by std)
scores = df_raw['Score'].values
mean_score = scores.mean()
std_score = scores.std()
normalized = (scores - mean_score) / std_score  # Broadcasting
print("Normalized scores (mean=0, std=1):")
print(normalized)
print("\n")

# ========== 16. Complete workflow for ML ==========
print("=== Complete ML Preprocessing Workflow ===")
# Select features
X = df_raw[['Age', 'Score']].values
print("Features (X) as NumPy array:")
print(X)
print("Shape:", X.shape)
print("\n")

# Create target (example: Pass if Score > 85)
y = (df_raw['Score'] > 85).values.astype(int)
print("Target (y) as NumPy array:")
print(y)
print("Shape:", y.shape)
print("\n")

print("Ready for: model.fit(X, y)")