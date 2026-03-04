# ========== COMPLETE SCIKIT-LEARN REFERENCE ==========
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            mean_squared_error, mean_absolute_error, r2_score)
import pickle
from joblib import dump, load
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("SCIKIT-LEARN COMPLETE REFERENCE - ALL ML CONCEPTS")
print("="*70)
print("\n")

# ========== 1. PREPARING DATA - X and y ==========
print("1. PREPARING DATA - Features (X) and Target (y)")
print("-"*70)

# Create sample dataset
data = {
    'age': [22, 25, 30, 35, 40, 45, 50, 55, 60, 65],
    'salary': [25000, 35000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000],
    'experience': [1, 2, 5, 7, 10, 12, 15, 18, 20, 22],
    'purchased': [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  # Target: 0=No, 1=Yes
}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)
print()

# Prepare X (features) and y (target)
X = df[['age', 'salary', 'experience']].values  # NumPy array
y = df['purchased'].values                       # NumPy array

print("X (Features) shape:", X.shape)  # (10, 3) → 10 samples, 3 features
print("y (Target) shape:", y.shape)    # (10,) → 10 samples
print("\nFirst 3 rows of X:")
print(X[:3])
print("\nFirst 5 values of y:", y[:5])
print("\n")

# ========== 2. TRAIN-TEST SPLIT ==========
print("2. TRAIN-TEST SPLIT - Separate data for training and testing")
print("-"*70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,      # 30% for testing, 70% for training
    random_state=42     # Fixed seed for reproducibility
)

print(f"Original dataset: {len(X)} samples")
print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.0f}%)")
print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.0f}%)")
print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print("\n")

# ========== 3. CLASSIFICATION - LOGISTIC REGRESSION ==========
print("3. CLASSIFICATION - Logistic Regression")
print("-"*70)

# Create and train model
log_model = LogisticRegression(random_state=42)
log_model.fit(X_train, y_train)

# Make predictions
y_pred_log = log_model.predict(X_test)

# Evaluate
accuracy_log = accuracy_score(y_test, y_pred_log)
print(f"Accuracy: {accuracy_log * 100:.2f}%")

# Confusion Matrix
cm_log = confusion_matrix(y_test, y_pred_log)
print("\nConfusion Matrix:")
print(cm_log)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))

# Visualize predictions
plt.figure(figsize=(8, 5))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual', s=100, alpha=0.6)
plt.scatter(range(len(y_pred_log)), y_pred_log, color='red', label='Predicted', 
            marker='x', s=100, linewidths=3)
plt.title('Logistic Regression: Actual vs Predicted', fontsize=14, fontweight='bold')
plt.xlabel('Sample Index')
plt.ylabel('Class (0 or 1)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('ml_1_logistic_regression.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

print("✓ Saved as 'ml_1_logistic_regression.png'\n")

# ========== 4. CLASSIFICATION - DECISION TREE ==========
print("4. CLASSIFICATION - Decision Tree")
print("-"*70)

dt_model = DecisionTreeClassifier(random_state=42, max_depth=3)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt * 100:.2f}%")
print("\n")

# ========== 5. CLASSIFICATION - RANDOM FOREST ==========
print("5. CLASSIFICATION - Random Forest")
print("-"*70)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf * 100:.2f}%")

# Feature importance
feature_importance = rf_model.feature_importances_
features = ['age', 'salary', 'experience']

plt.figure(figsize=(8, 5))
plt.barh(features, feature_importance, color='green', alpha=0.7)
plt.title('Random Forest - Feature Importance', fontsize=14, fontweight='bold')
plt.xlabel('Importance')
plt.grid(axis='x', alpha=0.3)
plt.savefig('ml_2_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

print("✓ Saved as 'ml_2_feature_importance.png'\n")

# ========== 6. CLASSIFICATION - KNN ==========
print("6. CLASSIFICATION - K-Nearest Neighbors")
print("-"*70)

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"KNN Accuracy: {accuracy_knn * 100:.2f}%")
print("\n")

# ========== 7. MODEL COMPARISON ==========
print("7. MODEL COMPARISON - Compare all classifiers")
print("-"*70)

models = {
    'Logistic Regression': accuracy_log,
    'Decision Tree': accuracy_dt,
    'Random Forest': accuracy_rf,
    'KNN': accuracy_knn
}

print("Model Performance:")
for name, acc in models.items():
    print(f"{name:20s}: {acc*100:.2f}%")

# Visualize comparison
plt.figure(figsize=(10, 6))
model_names = list(models.keys())
accuracies = [acc * 100 for acc in models.values()]
colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

bars = plt.bar(model_names, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.title('Classification Model Comparison', fontsize=16, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12)
plt.ylim(0, 105)
plt.xticks(rotation=15)
plt.grid(axis='y', alpha=0.3)
plt.savefig('ml_3_model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

print("\n✓ Saved as 'ml_3_model_comparison.png'\n")

# ========== 8. REGRESSION - LINEAR REGRESSION ==========
print("8. REGRESSION - Linear Regression (Predict continuous values)")
print("-"*70)

# Create regression dataset
X_reg = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y_reg = np.array([45, 55, 60, 68, 70, 78, 82, 88, 92, 98])

# Split
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Train
lin_model = LinearRegression()
lin_model.fit(X_reg_train, y_reg_train)

# Predict
y_reg_pred = lin_model.predict(X_reg_test)

# Evaluate
mse = mean_squared_error(y_reg_test, y_reg_pred)
mae = mean_absolute_error(y_reg_test, y_reg_pred)
r2 = r2_score(y_reg_test, y_reg_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.4f}")

print("\nActual vs Predicted:")
for actual, predicted in zip(y_reg_test, y_reg_pred):
    print(f"Actual: {actual:.2f}, Predicted: {predicted:.2f}, Difference: {abs(actual-predicted):.2f}")

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X_reg_train, y_reg_train, color='blue', s=100, label='Training Data', alpha=0.6)
plt.scatter(X_reg_test, y_reg_test, color='green', s=100, label='Test Data (Actual)', alpha=0.6)
plt.scatter(X_reg_test, y_reg_pred, color='red', s=100, marker='X', 
            label='Test Data (Predicted)', linewidths=2)

# Plot regression line
X_line = np.linspace(0, 11, 100).reshape(-1, 1)
y_line = lin_model.predict(X_line)
plt.plot(X_line, y_line, color='red', linewidth=2, linestyle='--', label='Regression Line')

plt.title('Linear Regression: Study Hours vs Exam Score', fontsize=14, fontweight='bold')
plt.xlabel('Study Hours', fontsize=12)
plt.ylabel('Exam Score', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('ml_4_linear_regression.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

print("\n✓ Saved as 'ml_4_linear_regression.png'\n")

# ========== 9. FEATURE SCALING - STANDARDSCALER ==========
print("9. FEATURE SCALING - StandardScaler (Mean=0, Std=1)")
print("-"*70)

# Original data
print("Original X_train (first 3 rows):")
print(X_train[:3])

# Create and fit scaler
scaler_std = StandardScaler()
scaler_std.fit(X_train)  # Fit ONLY on training data

# Transform
X_train_scaled = scaler_std.transform(X_train)
X_test_scaled = scaler_std.transform(X_test)

print("\nScaled X_train (first 3 rows):")
print(X_train_scaled[:3])
print(f"\nMean of scaled data: {X_train_scaled.mean(axis=0)}")
print(f"Std of scaled data: {X_train_scaled.std(axis=0)}")
print("\n")

# Train model with scaled data
log_model_scaled = LogisticRegression(random_state=42)
log_model_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = log_model_scaled.predict(X_test_scaled)

accuracy_scaled = accuracy_score(y_test, y_pred_scaled)
print(f"Accuracy with scaled features: {accuracy_scaled * 100:.2f}%")
print("\n")

# ========== 10. FEATURE SCALING - MINMAXSCALER ==========
print("10. FEATURE SCALING - MinMaxScaler (Scale to 0-1)")
print("-"*70)

scaler_minmax = MinMaxScaler()
scaler_minmax.fit(X_train)

X_train_minmax = scaler_minmax.transform(X_train)
X_test_minmax = scaler_minmax.transform(X_test)

print("MinMax Scaled X_train (first 3 rows):")
print(X_train_minmax[:3])
print(f"\nMin of scaled data: {X_train_minmax.min(axis=0)}")
print(f"Max of scaled data: {X_train_minmax.max(axis=0)}")
print("\n")

# ========== 11. CROSS-VALIDATION ==========
print("11. CROSS-VALIDATION - More reliable evaluation")
print("-"*70)

# 5-fold cross-validation
cv_scores = cross_val_score(log_model, X, y, cv=5)

print("Cross-Validation Scores (5 folds):", cv_scores)
print(f"Mean Accuracy: {cv_scores.mean() * 100:.2f}%")
print(f"Standard Deviation: {cv_scores.std() * 100:.2f}%")

# Visualize CV scores
plt.figure(figsize=(8, 5))
folds = [f'Fold {i+1}' for i in range(5)]
plt.bar(folds, cv_scores * 100, color='purple', alpha=0.7, edgecolor='black')
plt.axhline(cv_scores.mean() * 100, color='red', linestyle='--', 
            linewidth=2, label=f'Mean = {cv_scores.mean()*100:.2f}%')
plt.title('5-Fold Cross-Validation Results', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12)
plt.ylim(0, 105)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.savefig('ml_5_cross_validation.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

print("\n✓ Saved as 'ml_5_cross_validation.png'\n")

# ========== 12. HANDLING CATEGORICAL DATA - LABEL ENCODING ==========
print("12. LABEL ENCODING - Convert categories to numbers")
print("-"*70)

# Sample categorical data
colors = np.array(['red', 'blue', 'green', 'blue', 'red', 'green', 'red'])

# Encode
label_encoder = LabelEncoder()
colors_encoded = label_encoder.fit_transform(colors)

print("Original:", colors)
print("Encoded:", colors_encoded)  # [2 0 1 0 2 1 2]

# Decode back
colors_decoded = label_encoder.inverse_transform(colors_encoded)
print("Decoded:", colors_decoded)

# Show mapping
print("\nMapping:")
for i, color in enumerate(label_encoder.classes_):
    print(f"{color} → {i}")
print("\n")

# ========== 13. CONFUSION MATRIX VISUALIZATION ==========
print("13. CONFUSION MATRIX VISUALIZATION")
print("-"*70)

# Create larger dataset for better confusion matrix
np.random.seed(42)
X_large = np.random.randn(100, 3)
y_large = (X_large[:, 0] + X_large[:, 1] > 0).astype(int)

X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
    X_large, y_large, test_size=0.3, random_state=42
)

model_cm = LogisticRegression()
model_cm.fit(X_train_l, y_train_l)
y_pred_cm = model_cm.predict(X_test_l)

cm = confusion_matrix(y_test_l, y_pred_cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap='Blues', aspect='auto')
plt.colorbar(label='Count')
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)

# Add text annotations
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', 
                color='white' if cm[i, j] > cm.max()/2 else 'black',
                fontsize=20, fontweight='bold')

plt.xticks([0, 1], ['Class 0', 'Class 1'])
plt.yticks([0, 1], ['Class 0', 'Class 1'])
plt.savefig('ml_6_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

print("Confusion Matrix:")
print(cm)
print("\n✓ Saved as 'ml_6_confusion_matrix.png'\n")

# ========== 14. SAVE AND LOAD MODEL ==========
print("14. SAVING AND LOADING MODELS")
print("-"*70)

# Train a model
model_to_save = RandomForestClassifier(n_estimators=50, random_state=42)
model_to_save.fit(X_train, y_train)

# Save with pickle
with open('model_pickle.pkl', 'wb') as file:
    pickle.dump(model_to_save, file)
print("✓ Model saved as 'model_pickle.pkl' (using pickle)")

# Save with joblib (recommended for large models)
dump(model_to_save, 'model_joblib.pkl')
print("✓ Model saved as 'model_joblib.pkl' (using joblib)")

# Load with pickle
with open('model_pickle.pkl', 'rb') as file:
    loaded_model_pickle = pickle.load(file)
print("✓ Model loaded from pickle")

# Load with joblib
loaded_model_joblib = load('model_joblib.pkl')
print("✓ Model loaded from joblib")

# Test loaded model
test_prediction = loaded_model_joblib.predict(X_test)
test_accuracy = accuracy_score(y_test, test_prediction)
print(f"Loaded model accuracy: {test_accuracy * 100:.2f}%")
print("\n")

# ========== 15. COMPLETE ML PIPELINE ==========
print("15. COMPLETE ML PIPELINE - End-to-end workflow")
print("-"*70)

# Step 1: Load data
pipeline_data = {
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
    'feature3': np.random.randn(100),
    'target': np.random.randint(0, 2, 100)
}
pipeline_df = pd.DataFrame(pipeline_data)

# Step 2: Prepare X and y
X_pipeline = pipeline_df[['feature1', 'feature2', 'feature3']].values
y_pipeline = pipeline_df['target'].values

# Step 3: Split
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_pipeline, y_pipeline, test_size=0.2, random_state=42
)

# Step 4: Scale
scaler_p = StandardScaler()
scaler_p.fit(X_train_p)
X_train_p_scaled = scaler_p.transform(X_train_p)
X_test_p_scaled = scaler_p.transform(X_test_p)

# Step 5: Train
model_p = RandomForestClassifier(n_estimators=100, random_state=42)
model_p.fit(X_train_p_scaled, y_train_p)

# Step 6: Predict
y_pred_p = model_p.predict(X_test_p_scaled)

# Step 7: Evaluate
accuracy_p = accuracy_score(y_test_p, y_pred_p)
precision_p = precision_score(y_test_p, y_pred_p)
recall_p = recall_score(y_test_p, y_pred_p)
f1_p = f1_score(y_test_p, y_pred_p)

print("Complete Pipeline Results:")
print(f"Accuracy:  {accuracy_p * 100:.2f}%")
print(f"Precision: {precision_p * 100:.2f}%")
print(f"Recall:    {recall_p * 100:.2f}%")
print(f"F1-Score:  {f1_p * 100:.2f}%")

# Step 8: Save everything
dump(model_p, 'pipeline_model.pkl')
dump(scaler_p, 'pipeline_scaler.pkl')
print("\n✓ Model and scaler saved")
print("\n")

# ========== 16. FINAL DASHBOARD ==========
print("16. COMPLETE ML DASHBOARD")
print("-"*70)

fig = plt.figure(figsize=(16, 12))

# Plot 1: Model comparison
ax1 = plt.subplot(2, 3, 1)
model_names = ['Logistic\nRegression', 'Decision\nTree', 'Random\nForest', 'KNN']
accuracies_list = [accuracy_log, accuracy_dt, accuracy_rf, accuracy_knn]
colors_list = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
bars = ax1.bar(model_names, [a*100 for a in accuracies_list], color=colors_list, alpha=0.8)
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
ax1.set_title('Model Accuracy Comparison', fontsize=13, fontweight='bold')
ax1.set_ylabel('Accuracy (%)')
ax1.set_ylim(0, 105)
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Feature importance
ax2 = plt.subplot(2, 3, 2)
ax2.barh(features, feature_importance, color='green', alpha=0.7)
ax2.set_title('Feature Importance', fontsize=13, fontweight='bold')
ax2.set_xlabel('Importance')
ax2.grid(axis='x', alpha=0.3)

# Plot 3: Cross-validation
ax3 = plt.subplot(2, 3, 3)
folds = [f'F{i+1}' for i in range(5)]
ax3.bar(folds, cv_scores * 100, color='purple', alpha=0.7)
ax3.axhline(cv_scores.mean() * 100, color='red', linestyle='--', linewidth=2)
ax3.set_title('Cross-Validation (5-Fold)', fontsize=13, fontweight='bold')
ax3.set_ylabel('Accuracy (%)')
ax3.set_ylim(0, 105)
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Confusion matrix
ax4 = plt.subplot(2, 3, 4)
im = ax4.imshow(cm, cmap='Blues')
ax4.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
ax4.set_xlabel('Predicted')
ax4.set_ylabel('Actual')
for i in range(2):
    for j in range(2):
        ax4.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > cm.max()/2 else 'black',
                fontsize=16, fontweight='bold')
ax4.set_xticks([0, 1])
ax4.set_yticks([0, 1])

# Plot 5: Linear regression
ax5 = plt.subplot(2, 3, 5)
ax5.scatter(X_reg, y_reg, color='blue', s=50, alpha=0.6, label='Data')
X_line = np.linspace(0, 11, 100).reshape(-1, 1)
y_line = lin_model.predict(X_line)
ax5.plot(X_line, y_line, color='red', linewidth=2, linestyle='--', label='Regression Line')
ax5.set_title('Linear Regression', fontsize=13, fontweight='bold')
ax5.set_xlabel('Study Hours')
ax5.set_ylabel('Exam Score')
ax5.legend(loc='upper left')
ax5.grid(True, alpha=0.3)

# Plot 6: Metrics summary
ax6 = plt.subplot(2, 3, 6)
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [accuracy_p*100, precision_p*100, recall_p*100, f1_p*100]
bars_metrics = ax6.barh(metrics_names, metrics_values, color='#2ecc71', alpha=0.8)
for i, v in enumerate(metrics_values):
    ax6.text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')
ax6.set_title('Pipeline Performance Metrics', fontsize=13, fontweight='bold')
ax6.set_xlabel('Score (%)')
ax6.set_xlim(0, 105)
ax6.grid(axis='x', alpha=0.3)

plt.suptitle('COMPLETE MACHINE LEARNING DASHBOARD', fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('ml_7_complete_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

print("✓ Saved as 'ml_7_complete_dashboard.png'\n")

# ========== SUMMARY ==========
print("="*70)
print("SUMMARY - ALL SCIKIT-LEARN CONCEPTS COVERED")
print("="*70)
print("""
FILES CREATED:
1. ml_1_logistic_regression.png    - Classification predictions
2. ml_2_feature_importance.png     - Random Forest feature importance
3. ml_3_model_comparison.png       - Compare 4 classifiers
4. ml_4_linear_regression.png      - Regression visualization
5. ml_5_cross_validation.png       - 5-fold CV results
6. ml_6_confusion_matrix.png       - Confusion matrix heatmap
7. ml_7_complete_dashboard.png     - Complete ML dashboard
8. model_pickle.pkl                - Saved model (pickle)
9. model_joblib.pkl                - Saved model (joblib)
10. pipeline_model.pkl             - Complete pipeline model
11. pipeline_scaler.pkl            - Complete pipeline scaler

CONCEPTS COVERED:
✓ Preparing X and y from DataFrame
✓ Train-test split (70/30)
✓ Classification algorithms:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
✓ Regression algorithms:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
✓ Evaluation metrics:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix
  - MSE, MAE, R² Score
✓ Feature scaling:
  - StandardScaler (mean=0, std=1)
  - MinMaxScaler (0-1 range)
✓ Cross-validation (5-fold)
✓ Label encoding (categorical → numerical)
✓ Model saving/loading (pickle & joblib)
✓ Feature importance
✓ Complete end-to-end pipeline

CRITICAL RULES:
1. Always convert DataFrame to NumPy: .values or .to_numpy()
2. Always split BEFORE scaling
3. Fit scaler ONLY on training data
4. Use same scaler for train and test
5. Use random_state for reproducibility
6. Evaluate on TEST data, not training data

TYPICAL WORKFLOW:
1. Load data (Pandas)
2. Prepare X and y (NumPy arrays)
3. Split data (train_test_split)
4. Scale features (StandardScaler)
5. Choose model (LogisticRegression, etc.)
6. Train model (model.fit)
7. Predict (model.predict)
8. Evaluate (accuracy_score, etc.)
9. Save model (pickle/joblib)

READY FOR ANY ML COMPETITION! 🏆
""")

print("="*70)
print("Type 'next' for Module 5: Streamlit")
print("="*70)