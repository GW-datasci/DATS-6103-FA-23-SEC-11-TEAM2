#%%
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# %%
#loading the dataset
# Printing the total values in the dataset and the few rows
df = pd.read_csv('fraud_oracle.csv')
print('The data constain ', len(df),' observations.')
df.head()


# %%
# Finding all the null values
print(df.isnull().sum())


# %%
df.describe()
df.dtypes
# %%
for column in df.columns:
    unique_values = df[column].unique()
    print(f"'{column}': {unique_values}\n")

# %%
# We can see that Month and Day of week claimed have 0 in them.
#1. Investigating that!
print('DayOfWeekClaimed has ', len(df.loc[(df['DayOfWeekClaimed']=='0')]), ' row(s) with a 0')
print('MonthClaimed has ',len(df.loc[(df['MonthClaimed']=='0')]),' row(s) with a 0') 
print(' ')

print(df.loc[(df['DayOfWeekClaimed']=='0')])
print(df.loc[(df['MonthClaimed']=='0')])

# %%
# They are both present in the same column, drop the column!
df2 = df.loc[df['DayOfWeekClaimed']!='0']
df2.reset_index(drop=True, inplace=True)
len(df2)

# %%
#Age can't be 0. But we have a value which is zero. Checking that out!
len(df2[df2['Age']==0])
#There are 319 rows containg 0 as age!
df2.loc[df2['Age']==0, 'AgeOfPolicyHolder']
#But it also says that all 319 rows Age of Policy holder is between 16-17 years!

# %%
df2['Age'].mean
#replacing all with the mean value = 16.5 for easier analysis!
df2.loc[df2['Age']==0,'Age']=16.5

# %%
#verifying the result for AGE! 
# print(df2['Age'].unique()==0)
# len(df2[df2['Age']==0])
print(len(df2.drop_duplicates())==len(df2))

# %%
# Count the instances of fraud and no fraud
fraud_counts = df2['FraudFound_P'].value_counts()

# Create a bar plot
sns.barplot(x=fraud_counts.index, y=fraud_counts.values)

# Adding titles and labels for clarity
plt.title('Distribution of Fraudulent and Non-Fraudulent Cases')
plt.xlabel('Fraud Found (0 = No Fraud, 1 = Fraud)')
plt.ylabel('Number of Cases')
plt.xticks(range(2), ['No Fraud', 'Fraud'])

# Show the plot
plt.show()
# %%
#Univariate analysis
# Identify numerical and categorical columns
numerical_cols = df2.select_dtypes(include=['int64', 'float64']).columns
numerical_cols
categorical_cols = df2.select_dtypes(include=['object', 'category']).columns
categorical_cols

# Plotting histograms for numerical columns
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df2[col], kde=False)
    plt.title(f'Distribution of {col}')
    plt.ylabel('Frequency')
    plt.show()

# Plotting bar charts for categorical columns
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(y=col, data=df2)
    plt.title(f'Distribution of {col}')
    plt.show()

# %%
#Bivariate plots
# Identify numerical and categorical columns
numerical_cols = df2.select_dtypes(include=['int64', 'float64']).columns
numerical_cols
categorical_cols = df2.select_dtypes(include=['object', 'category']).columns


#%%
# Plotting box plots for numerical columns
for col in numerical_cols:
    if col != 'FraudFound_P':  # Exclude the target variable itself
        plt.figure(figsize=(8, 4))
        sns.boxplot(x='FraudFound_P', y=col, data=df2)
        plt.title(f'{col} Distribution by Fraud Found')
        plt.xlabel('Fraud Found (0 = No Fraud, 1 = Fraud)')
        plt.show()

# Plotting grouped bar charts for categorical columns
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, hue='FraudFound_P', data=df2)
    plt.title(f'{col} Distribution by Fraud Found')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.legend(title='Fraud Found', labels=['No Fraud', 'Fraud'])
    plt.show()

# %%
# Pie chart for 'FraudFound_P'
fraud_counts = df2['FraudFound_P'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(fraud_counts, labels=['No Fraud', 'Fraud'], autopct='%1.1f%%', startangle=140)
plt.title('Fraud Distribution')
plt.show()

# %%
sns.lmplot(x='WeekOfMonthClaimed', y='PolicyNumber', hue='FraudFound_P', data=df2, aspect=1.5)
plt.title('Linear Relationship between Two Numerical Features')
plt.show()


#%%
# Select a subset of features to avoid a cluttered plot
subset_features = ['DayOfWeek', 'AccidentArea', 'FraudFound_P']  # Replace with actual feature names
sns.pairplot(df2[subset_features], hue='FraudFound_P')
plt.show()

# %%
#Correlation matrix
# Selecting numerical features
numerical_df = df2.select_dtypes(include=['int64', 'float64'])

# Compute the correlation matrix
corr_matrix = numerical_df.corr()
corr_matrix

# %%
# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix for Numerical Features')
plt.show()
# %%
#Chi test for categorical values
def perform_chi_square_test(df, column, target):
    contingency_table = pd.crosstab(df[column], df[target])
    chi2, p, dof, ex = chi2_contingency(contingency_table)
    return chi2, p

# Assuming 'FraudFound_P' is your target variable
target_variable = 'FraudFound_P'

# Looping through categorical columns and performing chi-square tests
for column in df2.select_dtypes(include=['object', 'category']).columns:
    chi2, p = perform_chi_square_test(df2, column, target_variable)
    print(f"Chi-Square Test for {column}:\nChi2 Statistic: {chi2}, P-value: {p}\n")
# %%
