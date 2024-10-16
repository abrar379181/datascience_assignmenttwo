import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load and merge datasets
df1 = pd.read_csv('dataset1.csv')
df2 = pd.read_csv('dataset2.csv')
df3 = pd.read_csv('dataset3.csv')
df = df1.merge(df2, on='ID').merge(df3, on='ID')

# Calculate total screen time
screen_time_cols = ['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']
df['total_screen_time'] = df[screen_time_cols].sum(axis=1)

# Calculate average well-being score 
well_being_cols = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 
                   'Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']
df['avg_well_being'] = df[well_being_cols].mean(axis=1)

# Prepare data for regression
X = df['avg_well_being'].values.reshape(-1, 1)
y = df['total_screen_time'].values

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Calculate R-squared
r2 = r2_score(y, y_pred)

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5)
plt.plot(X, y_pred, color='red', linewidth=2)

plt.title('Simple Linear Regression: Average Well-being vs Total Screen Time')
plt.xlabel('Average Well-being Score')
plt.ylabel('Total Screen Time (hours)')

# Add R-squared value and regression equation to the plot
equation = f'y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}'
plt.text(0.05, 0.95, f'R² = {r2:.4f}\n{equation}', transform=plt.gca().transAxes, 
         verticalalignment='top')

plt.tight_layout()
plt.show()

# Print additional information
print(f"Regression Equation: {equation}")
print(f"R-squared: {r2:.4f}")
print(f"Coefficient (slope): {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# Print summary statistics
print("\nSummary Statistics:")
print(f"Average Well-being Score: Mean = {df['avg_well_being'].mean():.2f}, Std = {df['avg_well_being'].std():.2f}")
print(f"Total Screen Time: Mean = {df['total_screen_time'].mean():.2f}, Std = {df['total_screen_time'].std():.2f}")