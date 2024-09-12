import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Load the datasets
df1 = pd.read_csv('dataset1.csv')
df2 = pd.read_csv('dataset2.csv')
df3 = pd.read_csv('dataset3.csv')

# Merge the datasets
df = df1.merge(df2, on='ID').merge(df3, on='ID')

# Inferential Statistical Analysis 1: T-test for screen time difference between deprived and non-deprived areas
def screen_time_ttest_by_deprivation():
    df['total_screen_time'] = df['C_we'] + df['C_wk'] + df['G_we'] + df['G_wk'] + df['S_we'] + df['S_wk'] + df['T_we'] + df['T_wk']
    deprived = df[df['deprived'] == 1]['total_screen_time']
    non_deprived = df[df['deprived'] == 0]['total_screen_time']
    
    t_stat, p_value = stats.ttest_ind(deprived, non_deprived)
    
    return t_stat, p_value

# Inferential Statistical Analysis 2: Correlation between total screen time and average well-being score
def screen_time_wellbeing_correlation():
    df['total_screen_time'] = df['C_we'] + df['C_wk'] + df['G_we'] + df['G_wk'] + df['S_we'] + df['S_wk'] + df['T_we'] + df['T_wk']
    wellbeing_columns = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']
    df['avg_wellbeing'] = df[wellbeing_columns].mean(axis=1)
    
    correlation, p_value = stats.pearsonr(df['total_screen_time'], df['avg_wellbeing'])
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df['total_screen_time'], df['avg_wellbeing'], alpha=0.5)
    plt.title('Total Screen Time vs Average Well-being Score')
    plt.xlabel('Total Screen Time (hours)')
    plt.ylabel('Average Well-being Score')
    plt.show()
    
    return correlation, p_value

# Run the analyses

print("\nInferential Analysis 1:")
t_stat, p_value = screen_time_ttest_by_deprivation()
print(f"T-statistic: {t_stat}, p-value: {p_value}")

print("\nInferential Analysis 2:")
correlation, p_value = screen_time_wellbeing_correlation()
print(f"Correlation: {correlation}, p-value: {p_value}")