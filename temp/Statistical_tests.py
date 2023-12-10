#%%[markdown]

import pandas as pd

file_path = 'fraud_oracle.csv'

# Load the dataset into a Pandas DataFrame
df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame to check if it's loaded correctly
df.head()


# %%[markdown]
## Statistical Tests

## SMART Question:

#Is there a significant difference in the mean age between insurance claims with and without fraud, and does this difference vary based on the reported driver rating?
# %% [markdown]
## T-Test

#An independent samples t-test to compare the mean ages between insurance claims with and without fraud. 
#
#**Null Hypothesis (H0):** There is no significant difference in the mean age between claims with fraud ("FraudFound_P" = 1) and claims without fraud ("FraudFound_P" = 0).
#
#**Alternative Hypothesis (H1):** There is a significant difference in the mean age between claims with fraud and claims without fraud.
#Now, let's perform the t-test using Python and the scipy library:
# %% [markdown]
#
import scipy.stats as stats

# Assuming df is your DataFrame
fraud_age = df[df['FraudFound_P'] == 1]['Age']
non_fraud_age = df[df['FraudFound_P'] == 0]['Age']

# Perform t-test
t_stat, p_value = stats.ttest_ind(fraud_age, non_fraud_age, equal_var=False)

# Print results
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")
#
# Interpretation
alpha = 0.05  # Significance level
#
if p_value < alpha:
    print("Reject the null hypothesis.")
    print("There is a significant difference in the mean age between claims with fraud and claims without fraud.")
else:
    print("Fail to reject the null hypothesis.")
    print("There is no significant difference in the mean age between claims with fraud and claims without fraud.")


# %% [markdown]
#The analysis reveals a significant age difference between insurance claims with and without fraud. This indicates that age is a potential key factor in identifying fraudulent claims. Understanding these age patterns can inform targeted fraud detection strategies, emphasizing the importance of considering demographic factors in risk assessment and mitigation within the insurance industry.

# %% [markdown]

## Two-way ANOVA test

#The hypotheses for the interaction effect in the context of the two-way ANOVA:
#
#**Null Hypothesis (H0):**
#There is no interaction effect between fraud (fraud vs. non-fraud) and driver ratings on the mean age.
#
#**Alternate Hypothesis (H1):**
#There is an interaction effect between fraud and driver ratings on the mean age.

# %% [markdown]

import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


# Filter the data for fraud and non-fraud claims
fraud_data = df[df['FraudFound_P'] == 1]
non_fraud_data = df[df['FraudFound_P'] == 0]

# Perform a two-way ANOVA
formula = 'Age ~ C(FraudFound_P) * C(DriverRating)'
model = ols(formula, df).fit()
anova_table = anova_lm(model, typ=2)

# Print the ANOVA table
print(anova_table)

# Interpretation of the results
p_value_interaction = anova_table['PR(>F)']['C(FraudFound_P):C(DriverRating)']

# Set the significance level
alpha = 0.05

# Print the p-value for the interaction effect
print(f"P-value for Interaction Effect: {p_value_interaction}")

# Interpret the results
if p_value_interaction < alpha:
    print("Reject the null hypothesis.")
    print("There is a significant interaction effect between fraud and driver ratings on the mean age.")
else:
    print("Fail to reject the null hypothesis.")
    print("There is not enough evidence to claim a significant interaction effect.")


# %% [markdown]

# Answer to SMART Question: 
#
#Based on the results of the two-way ANOVA test:

#1. **Difference in Mean Age between Fraud and Non-Fraud Claims:**
#
# - There is a significant difference in the mean age between insurance claims with and without fraud (p-value < 0.05). This suggests that age is a relevant factor in distinguishing between fraudulent and non-fraudulent claims.

#2. **Variation in Mean Age Based on Driver Rating:**
#   - There is no significant difference in the mean age across different levels of reported driver ratings (p-value > 0.05). This indicates that, overall, reported driver ratings do not contribute significantly to variations in mean age.

#3. **Interaction Effect between Fraud and Driver Rating on Mean Age:**
#  - The analysis does not provide enough evidence to support a significant interaction effect between fraud and driver ratings on the mean age (p-value > 0.05). This suggests that the impact of fraud on mean age does not vary based on the reported driver rating, and vice versa.

#**Conclusion:**
#In summary, age appears to be a relevant factor in identifying fraudulent claims, as there is a significant difference in mean age between fraud and non-fraud claims. However, the reported driver rating does not contribute significantly to variations in mean age, and there is no significant interaction effect between fraud and driver ratings on mean age. Therefore, while age is a distinguishing factor, the reported driver rating does not appear to modify the relationship between age and fraud in a statistically significant manner.

# %%
