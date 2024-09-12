# inferential statistics investigation

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import ttest_ind

dataset1 = pd.read_csv('dataset1.csv')
dataset2 = pd.read_csv('dataset2.csv')
dataset3 = pd.read_csv('dataset3.csv')

print(dataset1.head())
print(dataset2.head())
print(dataset3.head())

merged_data = dataset1.merge(dataset2, on='ID').merge(dataset3, on='ID')
print(merged_data.head())


# screen time
merged_data['total_screen_time'] = merged_data[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']].sum(axis=1)

# Investigation two
# Correlation analysis
corrs = merged_data[['total_screen_time'] + list(merged_data.columns[merged_data.columns.str.startswith('Optm')])].corr()
print(corrs)

# Visualization
sns.heatmap(corrs, annot=True)
plt.show()

# Linear regression
import statsmodels.api as sm

X = merged_data['total_screen_time']
Y = merged_data['Optm']
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
print(model.summary())

# Investigation two
# T-Test

grouped_deprivation = merged_data.groupby('deprived')['total_screen_time'].mean()
print(grouped_deprivation)

t_stat, p_value = ttest_ind(merged_data[merged_data['deprived'] == 1]['total_screen_time'],
                            merged_data[merged_data['deprived'] == 0]['total_screen_time'])

print(f'T-Test between deprived and non-deprived people: t={t_stat}, p={p_value}')