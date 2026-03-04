# ========== COMPLETE MATPLOTLIB REFERENCE ==========
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("="*60)
print("MATPLOTLIB COMPLETE REFERENCE - ALL CHART TYPES")
print("="*60)
print("\n")

# ========== 1. BASIC LINE PLOT ==========
print("1. BASIC LINE PLOT - Show trends")
print("-"*60)

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

plt.figure(figsize=(8, 5))
plt.plot(x, y)
plt.title('Basic Line Plot')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.grid()
plt.savefig('1_basic_line.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

print("✓ Saved as '1_basic_line.png'\n")

# ========== 2. STYLED LINE PLOT ==========
print("2. STYLED LINE PLOT - With markers and styling")
print("-"*60)

days = np.array([1, 2, 3, 4, 5, 6, 7])
sales = np.array([100, 120, 115, 140, 160, 155, 180])

plt.figure(figsize=(8, 5))
plt.plot(days, sales, marker='o', linestyle='-', color='blue', 
         linewidth=2, markersize=8, label='Daily Sales')
plt.title('Weekly Sales Trend', fontsize=16, fontweight='bold')
plt.xlabel('Day', fontsize=12)
plt.ylabel('Sales ($)', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('2_styled_line.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

print("✓ Saved as '2_styled_line.png'\n")

# ========== 3. MULTIPLE LINES ON ONE PLOT ==========
print("3. MULTIPLE LINES - Compare trends")
print("-"*60)

x = np.array([1, 2, 3, 4, 5])
revenue = np.array([100, 150, 140, 180, 200])
costs = np.array([50, 60, 65, 70, 75])
profit = revenue - costs

plt.figure(figsize=(10, 6))
plt.plot(x, revenue, marker='o', label='Revenue', color='green', linewidth=2)
plt.plot(x, costs, marker='s', label='Costs', color='red', linewidth=2)
plt.plot(x, profit, marker='^', label='Profit', color='blue', linewidth=2)

plt.title('Business Metrics Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Amount ($)', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig('3_multiple_lines.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

print("✓ Saved as '3_multiple_lines.png'\n")

# ========== 4. BAR CHART (VERTICAL) ==========
print("4. BAR CHART - Compare categories")
print("-"*60)

products = ['Product A', 'Product B', 'Product C', 'Product D']
revenue = np.array([5000, 7000, 6000, 8000])

plt.figure(figsize=(8, 5))
bars = plt.bar(products, revenue, color=['blue', 'green', 'orange', 'red'], 
               alpha=0.7, edgecolor='black')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'${height:,.0f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title('Revenue by Product', fontsize=16, fontweight='bold')
plt.xlabel('Product', fontsize=12)
plt.ylabel('Revenue ($)', fontsize=12)
plt.ylim(0, 9000)
plt.grid(axis='y', alpha=0.3)
plt.savefig('4_bar_chart.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

print("✓ Saved as '4_bar_chart.png'\n")

# ========== 5. HORIZONTAL BAR CHART ==========
print("5. HORIZONTAL BAR CHART - Alternative view")
print("-"*60)

categories = ['Category A', 'Category B', 'Category C', 'Category D']
values = np.array([45, 67, 52, 78])

plt.figure(figsize=(8, 5))
plt.barh(categories, values, color='purple', alpha=0.7, edgecolor='black')
plt.title('Performance by Category', fontsize=16, fontweight='bold')
plt.xlabel('Score', fontsize=12)
plt.ylabel('Category', fontsize=12)
plt.grid(axis='x', alpha=0.3)
plt.savefig('5_horizontal_bar.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

print("✓ Saved as '5_horizontal_bar.png'\n")

# ========== 6. SCATTER PLOT ==========
print("6. SCATTER PLOT - Show relationships")
print("-"*60)

study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
exam_scores = np.array([50, 55, 60, 65, 70, 75, 78, 82, 85, 90])

plt.figure(figsize=(8, 5))
plt.scatter(study_hours, exam_scores, color='red', s=100, alpha=0.6, 
            edgecolors='black', linewidths=1)
plt.title('Study Hours vs Exam Scores', fontsize=16, fontweight='bold')
plt.xlabel('Study Hours', fontsize=12)
plt.ylabel('Exam Score', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('6_scatter_plot.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

print("✓ Saved as '6_scatter_plot.png'\n")

# ========== 7. SCATTER PLOT WITH VARYING SIZES AND COLORS ==========
print("7. ADVANCED SCATTER - Size and color variations")
print("-"*60)

x = np.random.rand(50) * 100
y = np.random.rand(50) * 100
sizes = np.random.rand(50) * 500  # Varying sizes
colors = np.random.rand(50)  # Varying colors

plt.figure(figsize=(8, 6))
scatter = plt.scatter(x, y, s=sizes, c=colors, alpha=0.5, 
                     cmap='viridis', edgecolors='black', linewidths=0.5)
plt.colorbar(scatter, label='Color Scale')
plt.title('Advanced Scatter Plot', fontsize=16, fontweight='bold')
plt.xlabel('X Axis', fontsize=12)
plt.ylabel('Y Axis', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('7_advanced_scatter.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

print("✓ Saved as '7_advanced_scatter.png'\n")

# ========== 8. HISTOGRAM - SINGLE DATASET ==========
print("8. HISTOGRAM - Show distribution")
print("-"*60)

# Generate random ages (normal distribution)
ages = np.random.normal(30, 10, 1000)  # Mean=30, std=10, 1000 samples

plt.figure(figsize=(8, 5))
plt.hist(ages, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Age Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Age', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.axvline(ages.mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean = {ages.mean():.1f}')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.savefig('8_histogram.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

print("✓ Saved as '8_histogram.png'\n")

# ========== 9. MULTIPLE HISTOGRAMS ==========
print("9. OVERLAPPING HISTOGRAMS - Compare distributions")
print("-"*60)

group_a = np.random.normal(60, 10, 500)
group_b = np.random.normal(70, 15, 500)

plt.figure(figsize=(8, 5))
plt.hist(group_a, bins=20, alpha=0.5, label='Group A', color='blue', edgecolor='black')
plt.hist(group_b, bins=20, alpha=0.5, label='Group B', color='orange', edgecolor='black')
plt.title('Score Distribution Comparison', fontsize=16, fontweight='bold')
plt.xlabel('Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(loc='upper right')
plt.grid(axis='y', alpha=0.3)
plt.savefig('9_overlapping_histogram.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

print("✓ Saved as '9_overlapping_histogram.png'\n")

# ========== 10. PIE CHART ==========
print("10. PIE CHART - Show proportions")
print("-"*60)

categories = ['Marketing', 'Development', 'Sales', 'Support']
expenses = np.array([30, 40, 20, 10])
colors = ['gold', 'lightblue', 'lightgreen', 'pink']
explode = (0.1, 0, 0, 0)  # Explode first slice

plt.figure(figsize=(8, 6))
plt.pie(expenses, labels=categories, autopct='%1.1f%%', startangle=90,
        colors=colors, explode=explode, shadow=True)
plt.title('Budget Allocation', fontsize=16, fontweight='bold')
plt.savefig('10_pie_chart.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

print("✓ Saved as '10_pie_chart.png'\n")

# ========== 11. SUBPLOTS (2x2 GRID) ==========
print("11. SUBPLOTS - Multiple charts in one figure")
print("-"*60)

x = np.linspace(0, 10, 100)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Line
axes[0, 0].plot(x, np.sin(x), color='blue')
axes[0, 0].set_title('Sine Wave', fontweight='bold')
axes[0, 0].set_xlabel('X')
axes[0, 0].set_ylabel('sin(x)')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Cosine
axes[0, 1].plot(x, np.cos(x), color='red')
axes[0, 1].set_title('Cosine Wave', fontweight='bold')
axes[0, 1].set_xlabel('X')
axes[0, 1].set_ylabel('cos(x)')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Bar
categories = ['A', 'B', 'C', 'D']
values = [25, 40, 30, 55]
axes[1, 0].bar(categories, values, color='green', alpha=0.7)
axes[1, 0].set_title('Bar Chart', fontweight='bold')
axes[1, 0].set_ylabel('Values')
axes[1, 0].grid(axis='y', alpha=0.3)

# Plot 4: Scatter
x_scatter = np.random.rand(50) * 10
y_scatter = np.random.rand(50) * 10
axes[1, 1].scatter(x_scatter, y_scatter, color='purple', alpha=0.6)
axes[1, 1].set_title('Scatter Plot', fontweight='bold')
axes[1, 1].set_xlabel('X')
axes[1, 1].set_ylabel('Y')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('11_subplots.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

print("✓ Saved as '11_subplots.png'\n")

# ========== 12. PLOTTING FROM PANDAS DATAFRAME ==========
print("12. PLOTTING FROM PANDAS - Direct DataFrame plotting")
print("-"*60)

# Create DataFrame
data = {
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    'Sales': [100, 120, 115, 140, 160, 155],
    'Profit': [20, 25, 22, 30, 35, 32],
    'Expenses': [80, 95, 93, 110, 125, 123]
}
df = pd.DataFrame(data)

# Plot 1: Line plot from Pandas
plt.figure(figsize=(10, 5))
df.plot(x='Month', y='Sales', kind='line', marker='o', color='blue', 
        legend=False, ax=plt.gca())
plt.title('Monthly Sales (Pandas)', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Sales ($)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('12_pandas_line.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

print("✓ Saved as '12_pandas_line.png'\n")

# Plot 2: Multiple columns
plt.figure(figsize=(10, 5))
df.plot(x='Month', y=['Sales', 'Profit', 'Expenses'], kind='line', 
        marker='o', ax=plt.gca())
plt.title('Business Metrics (Pandas)', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Amount ($)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('12_pandas_multiple.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

print("✓ Saved as '12_pandas_multiple.png'\n")

# Plot 3: Bar chart from Pandas
plt.figure(figsize=(10, 5))
df.plot(x='Month', y='Profit', kind='bar', color='green', 
        legend=False, ax=plt.gca())
plt.title('Monthly Profit (Pandas)', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Profit ($)', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.savefig('12_pandas_bar.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

print("✓ Saved as '12_pandas_bar.png'\n")

# ========== 13. BOX PLOT - Show statistical distribution ==========
print("13. BOX PLOT - Statistical distribution")
print("-"*60)

data_to_plot = [np.random.normal(100, 10, 200),
                np.random.normal(90, 20, 200),
                np.random.normal(80, 15, 200)]

plt.figure(figsize=(8, 6))
plt.boxplot(data_to_plot, labels=['Group A', 'Group B', 'Group C'],
            patch_artist=True, notch=True)
plt.title('Score Distribution by Group', fontsize=16, fontweight='bold')
plt.ylabel('Score', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.savefig('13_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

print("✓ Saved as '13_boxplot.png'\n")

# ========== 14. AREA PLOT ==========
print("14. AREA PLOT - Cumulative visualization")
print("-"*60)

x = np.arange(1, 8)
y1 = np.array([1, 2, 3, 4, 5, 6, 7])
y2 = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4])

plt.figure(figsize=(10, 5))
plt.fill_between(x, y1, alpha=0.5, label='Product A', color='blue')
plt.fill_between(x, y2, alpha=0.5, label='Product B', color='orange')
plt.title('Area Plot - Sales Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Day', fontsize=12)
plt.ylabel('Sales', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.savefig('14_area_plot.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

print("✓ Saved as '14_area_plot.png'\n")

# ========== 15. HEATMAP (using imshow) ==========
print("15. HEATMAP - 2D data visualization")
print("-"*60)

# Create correlation-like matrix
data_matrix = np.random.rand(5, 5)

plt.figure(figsize=(8, 6))
im = plt.imshow(data_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar(im, label='Value')
plt.title('Heatmap Example', fontsize=16, fontweight='bold')
plt.xlabel('Features', fontsize=12)
plt.ylabel('Samples', fontsize=12)

# Add text annotations
for i in range(5):
    for j in range(5):
        text = plt.text(j, i, f'{data_matrix[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=9)

plt.savefig('15_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

print("✓ Saved as '15_heatmap.png'\n")

# ========== 16. COMPLETE COMPETITION DASHBOARD ==========
print("16. COMPETITION DASHBOARD - Complete professional example")
print("-"*60)

# Create comprehensive data
models = ['Model A', 'Model B', 'Model C', 'Model D']
accuracy = np.array([85, 92, 88, 90])
training_time = np.array([120, 180, 150, 160])
f1_scores = np.array([0.83, 0.91, 0.86, 0.89])

# Prediction data
actual = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
predicted = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 1])

# Create 2x3 subplot grid
fig = plt.figure(figsize=(16, 10))

# Plot 1: Accuracy Comparison
ax1 = plt.subplot(2, 3, 1)
bars = ax1.bar(models, accuracy, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'], alpha=0.8)
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height}%', ha='center', va='bottom', fontweight='bold')
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=11)
ax1.set_ylim(80, 95)
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Training Time
ax2 = plt.subplot(2, 3, 2)
ax2.barh(models, training_time, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'], alpha=0.8)
ax2.set_title('Training Time', fontsize=14, fontweight='bold')
ax2.set_xlabel('Time (seconds)', fontsize=11)
ax2.grid(axis='x', alpha=0.3)

# Plot 3: F1 Score
ax3 = plt.subplot(2, 3, 3)
ax3.plot(models, f1_scores, marker='o', markersize=10, linewidth=2.5, color='#9b59b6')
ax3.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
ax3.set_ylabel('F1 Score', fontsize=11)
ax3.set_ylim(0.8, 0.95)
ax3.grid(True, alpha=0.3)

# Plot 4: Accuracy vs Time scatter
ax4 = plt.subplot(2, 3, 4)
scatter = ax4.scatter(training_time, accuracy, s=300, c=f1_scores, 
                     cmap='viridis', alpha=0.7, edgecolors='black', linewidths=2)
for i, model in enumerate(models):
    ax4.annotate(model, (training_time[i], accuracy[i]), 
                ha='center', va='bottom', fontweight='bold')
ax4.set_title('Accuracy vs Training Time', fontsize=14, fontweight='bold')
ax4.set_xlabel('Training Time (s)', fontsize=11)
ax4.set_ylabel('Accuracy (%)', fontsize=11)
ax4.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax4, label='F1 Score')

# Plot 5: Actual vs Predicted
ax5 = plt.subplot(2, 3, 5)
x_pos = np.arange(len(actual))
width = 0.35
ax5.bar(x_pos - width/2, actual, width, label='Actual', alpha=0.7, color='blue')
ax5.bar(x_pos + width/2, predicted, width, label='Predicted', alpha=0.7, color='orange')
ax5.set_title('Actual vs Predicted (Sample)', fontsize=14, fontweight='bold')
ax5.set_xlabel('Sample Index', fontsize=11)
ax5.set_ylabel('Class', fontsize=11)
ax5.legend()
ax5.grid(axis='y', alpha=0.3)

# Plot 6: Performance Metrics Summary
ax6 = plt.subplot(2, 3, 6)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
best_model_metrics = [92, 90, 91, 91]
ax6.barh(metrics, best_model_metrics, color='#2ecc71', alpha=0.8)
for i, v in enumerate(best_model_metrics):
    ax6.text(v + 1, i, str(v) + '%', va='center', fontweight='bold')
ax6.set_title('Best Model (Model B) Metrics', fontsize=14, fontweight='bold')
ax6.set_xlabel('Score (%)', fontsize=11)
ax6.set_xlim(85, 100)
ax6.grid(axis='x', alpha=0.3)

plt.suptitle('MACHINE LEARNING MODEL COMPARISON DASHBOARD', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('16_competition_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

print("✓ Saved as '16_competition_dashboard.png'\n")

# ========== SUMMARY ==========
print("="*60)
print("SUMMARY - ALL PLOTS CREATED AND SAVED")
print("="*60)
print("""
Files Created:
1.  1_basic_line.png              - Basic line plot
2.  2_styled_line.png             - Styled line with markers
3.  3_multiple_lines.png          - Multiple lines comparison
4.  4_bar_chart.png               - Vertical bar chart
5.  5_horizontal_bar.png          - Horizontal bar chart
6.  6_scatter_plot.png            - Basic scatter plot
7.  7_advanced_scatter.png        - Advanced scatter with colors
8.  8_histogram.png               - Distribution histogram
9.  9_overlapping_histogram.png   - Compare distributions
10. 10_pie_chart.png              - Pie chart for proportions
11. 11_subplots.png               - 2x2 subplot grid
12. 12_pandas_line.png            - Pandas line plot
13. 12_pandas_multiple.png        - Pandas multiple lines
14. 12_pandas_bar.png             - Pandas bar chart
15. 13_boxplot.png                - Box plot
16. 14_area_plot.png              - Area/fill plot
17. 15_heatmap.png                - Heatmap visualization
18. 16_competition_dashboard.png  - Complete dashboard

KEY MATPLOTLIB FUNCTIONS COVERED:
- plt.plot()        → Line plots
- plt.bar()         → Bar charts (vertical)
- plt.barh()        → Bar charts (horizontal)
- plt.scatter()     → Scatter plots
- plt.hist()        → Histograms
- plt.pie()         → Pie charts
- plt.boxplot()     → Box plots
- plt.fill_between()→ Area plots
- plt.imshow()      → Heatmaps
- plt.subplots()    → Multiple plots
- df.plot()         → Pandas direct plotting
- plt.savefig()     → Save plots
- plt.grid()        → Add grid
- plt.legend()      → Add legend
- plt.title()       → Add title
- plt.xlabel()      → X-axis label
- plt.ylabel()      → Y-axis label

READY FOR COMPETITION!
""")

print("="*60)
print("Type 'next' for Module 4: scikit-learn")
print("="*60)