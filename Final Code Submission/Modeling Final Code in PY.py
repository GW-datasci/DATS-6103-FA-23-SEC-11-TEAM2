#%%[markdown]

# # Insurance fraud detection

# # Data Dictionary
# 
# ## Categorical Variables
# 
# - **Month**: *object*
#   - Contains 3-letter abbreviations for the months of the year.
#   - Indicates the months in which the accident occurred.
# 
# - **WeekOfMonth**: *int64*
#   - Provides the week in the month the accident occurred.
# 
# - **DayOfWeek**: *object*
#   - Contains days of the week.
#   - Indicates the days of the week the accident occurred on.
# 
# - **Make**: *object*
#   - Contains a list of 19 car manufacturers.
# 
# - **AccidentArea**: *object*
#   - Classifies the area for the accident as "Urban" or "Rural".
# 
# - **DayOfWeekClaimed**: *object*
#   - Contains the day of the week the claim was filed.
#   - Also contains '0'; needs further investigation for missing data.
# 
# - **MonthClaimed**: *object*
#   - Contains 3-letter abbreviations for the months of the year.
#   - Contains '0'; needs further investigation for missing data.
# 
# - **WeekOfMonthClaimed**: *int64*
#   - Contains weeks in the month that the claim is filed.
# 
# - **Sex**: *object*
#   - Gender of the individual making the claim.
#   - Binary data, convert to 1 or 0.
# 
# - **MaritalStatus**: *object*
#   - Marital status of the individual making the claim.
# 
# - **Fault**: *object*
#   - Categorization of who was deemed at fault.
#   - Convert to binary, 1 or 0.
# 
# - **PolicyType**: *object*
#   - Contains two pieces of information:
#     - The type of insurance on the car (liability, all perils, collision).
#     - Category of the vehicle (sport, sedan, utility).
# 
# - **VehicleCategory**: *object*
#   - Contains the categorization of the vehicle (see PolicyType).
# 
# - **VehiclePrice**: *object*
#   - Contains ranges for the value of the vehicle.
#   - Replace ranges with the mean value of the range and convert to float.
# 
# - **PoliceReportFiled**: *object*
#   - Indicates whether a police report was filed for the accident.
#   - Convert to binary.
# 
# - **WitnessPresent**: *object*
#   - Indicates whether a witness was present.
#   - Convert to binary.
# 
# - **AgentType**: *object*
#   - Classifies an agent who is handling the claim as internal vs external.
#   - Convert to binary.
# 
# - **AddressChange_Claim**: *object*
#   - Guess: time from claim was filed to when the person moved (i.e., filed an address change).
#   - Replace each interval with the mean value of the range.
# 
# - **NumberOfCars**: *object*
#   - Guess: number of cars involved in the accident OR the number of cars covered under the policy.
#   - Replace each interval with the mean value of the range.
# 
# - **BasePolicy**: *object*
#   - Type of insurance coverage (see PolicyType).
# 
# ## Numeric Variables
# 
# - **Age**: *int64*
#   - Ages of individuals making claims.
#   - There is at least one individual with age 0; potential missing data.
# 
# - **Deductible**: *int64*
#   - The deductible amount (integer values).
# 
# - **DriverRating**: *int64*
#   - The scale is 1, 2, 3, 4.
#   - The name DriverRating implies the data is ordinal; further investigation needed.
# 
# - **PolicyNumber**: *int64*
#   - The masked policy number, appears to be the same as row number minus 1.
# 
# - **RepNumber**: *int64*
#   - Rep number is an integer from 1 - 16.
# 
# - **FraudFound_P**: *int64*
#   - Indicates whether the claim was fraudulent (1) or not (0); target variable.
# 
# - **Days_Policy_Accident**: *object*
#   - Guess: the number of days between when the policy was purchased and the accident occurred.
#   - Each value is a range of values; change these to be the mean of the range and make float.
# 
# - **Days_Policy_Claim**: *object*
#   - Guess: the number of days that pass between the policy was purchased and the claim was filed.
#   - Each value is a range; change these to be the mean of the ranges and make float.
# 
# - **PastNumberOfClaims**: *object*
#   - Previous number of claims filed by the policy holder (or claimant?).
# 
# - **AgeOfVehicle**: *object*
#   - Represents the age of the vehicle at the time of the accident.
#   - Each value is a range of years; change these to be the mean of the ranges and make float.
# 
# - **AgeOfPolicyHolder**: *object*
#   - Each value is a range of ages.
#   - Change these to be the mean of the ranges and make float.
# 
# - **NumberOfSupplements**: *object*
#   - Probably not the number of vitamins taken daily.
#   - Not sure what a supplement is in insurance.
# 
# - **NumberOfSupplements**: *object*
#   - Probably not the number of vitamins taken daily.
#   - Not sure what a supplement is in insurance.
# 
# - **NumberOfSuppliments**: *object*
#   - Probably not the number of vitamins taken daily.
#   - Not sure what a supplement is in insurance.
# 
# - **NumberOfSuppliments**: *object*
#   - Probably not the number of vitamins taken daily.
#   - Not sure what a supplement is in insurance.
#   
# - **NumberOfSuppliments**: *object*
#   - Probably not the number of vitamins taken daily.
#   - Not sure what a supplement is in insurance.
# 

#%%[markdown]


import pandas as pd

file_path = 'fraud_oracle.csv'

# Load the dataset into a Pandas DataFrame
df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame to check if it's loaded correctly
df.head()

#%%[markdown]


# Check unique values in the target variable
unique_values = df['FraudFound_P'].unique()
print("Unique values in the target variable (FraudFound_P):", unique_values)


#%%[markdown]


# Check for NaN values in the entire dataset
nan_values = df.isna().sum()
print("\nNaN values in the dataset:")
print(nan_values)


# Hence there are no missing values in the dataset

#%%[markdown]
import matplotlib.pyplot as plt
import seaborn as sns

# Count the occurrences of 0s and 1s in the target variable
fraud_counts = df['FraudFound_P'].value_counts()

# Plot the counts using a bar chart
plt.figure(figsize=(8, 6))
sns.countplot(x='FraudFound_P', data=df, palette='viridis')

# Annotate the bars with counts
for i, count in enumerate(fraud_counts):
    plt.text(i, count + 100, str(count), ha='center', va='bottom')

plt.title('Distribution of Fraudulent and Genuine Cases')
plt.xlabel('FraudFound_P (0: Genuine, 1: Fraudulent)')
plt.ylabel('Count')
plt.show()


#%%[markdown]


# Display the data types of each column
data_types = df.dtypes
print("Data types of each column:")
print(data_types)


#%%[markdown]


# Display unique values of each column
for column in df.columns:
    unique_values = df[column].unique()
    print(f"\nUnique values in '{column}':")
    print(unique_values)


#%%[markdown]


# Perform one-hot encoding for categorical variables
df_encoded = pd.get_dummies(df, columns=[
    'Month', 'DayOfWeek', 'Make', 'AccidentArea', 'DayOfWeekClaimed',
    'MonthClaimed', 'Sex', 'MaritalStatus', 'Fault', 'PolicyType',
    'VehicleCategory', 'VehiclePrice', 'Days_Policy_Accident',
    'Days_Policy_Claim', 'PastNumberOfClaims', 'AgeOfVehicle',
    'AgeOfPolicyHolder', 'PoliceReportFiled', 'WitnessPresent',
    'AgentType', 'NumberOfSuppliments', 'AddressChange_Claim',
    'NumberOfCars', 'BasePolicy'
], drop_first=True)

# Display the first few rows of the encoded DataFrame
df_encoded.head()


#%%[markdown]


#Check the new data dypes
data_types = df_encoded.dtypes
print("Data types of each column:")
print(data_types)


# ## Split the data into test and train
#%%[markdown]


#Split the data into test and train
from sklearn.model_selection import train_test_split

# Separate features (X) and target variable (y)
X = df_encoded.drop('FraudFound_P', axis=1)
y = df_encoded['FraudFound_P']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the resulting sets
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


#%%[markdown]


# Check for missing values in the training set
missing_values_train = X_train.isnull().sum()
print("Missing values in the training set:")
print(missing_values_train)


# ## Decision Tree classifier

#%%[markdown]


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

# Initialize Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)

# Train the Decision Tree model on the training set
dt_model.fit(X_train, y_train)

# Predictions on the testing set
y_pred_dt = dt_model.predict(X_test)

# Calculate AUC score for Decision Tree model
y_pred_proba_dt = dt_model.predict_proba(X_test)[:, 1]
auc_score_dt = roc_auc_score(y_test, y_pred_proba_dt)

# Evaluate the Decision Tree model
accuracy_dt = accuracy_score(y_test, y_pred_dt)
classification_report_result_dt = classification_report(y_test, y_pred_dt)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)

# Display evaluation metrics for Decision Tree model
print("Accuracy with Decision Tree:", accuracy_dt)
print("\nAUC Score with Decision Tree:", auc_score_dt)
print("\nClassification Report with Decision Tree:")
print(classification_report_result_dt)
print("\nConfusion Matrix with Decision Tree:")
print(conf_matrix_dt)

# Plot ROC Curve for Decision Tree model
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_proba_dt)
roc_auc_dt = auc(fpr_dt, tpr_dt)

plt.figure(figsize=(8, 6))
plt.plot(fpr_dt, tpr_dt, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc_dt))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Decision Tree)')
plt.legend(loc="lower right")
plt.show()


#%%[markdown]


## Hyperparameter tuning of Decision Tree classifier


#%%[markdown]


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

# Define the hyperparameter grid for Decision Tree
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV for Decision Tree
grid_search_dt = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid=param_grid_dt, scoring='roc_auc', cv=3)

# Perform Grid Search on the training set
grid_search_dt.fit(X_train, y_train)

# Display the best hyperparameters for Decision Tree
print("Best Hyperparameters (Decision Tree):", grid_search_dt.best_params_)

# Get the best Decision Tree model from the grid search
best_model_dt = grid_search_dt.best_estimator_

# Predictions on the testing set using the best model
y_pred_best_dt = best_model_dt.predict(X_test)
y_pred_proba_best_dt = best_model_dt.predict_proba(X_test)[:, 1]

# Calculate AUC score for the best Decision Tree model
auc_score_best_dt = roc_auc_score(y_test, y_pred_proba_best_dt)
print("\nAUC Score (Best Decision Tree):", auc_score_best_dt)

# Plot ROC Curve for the best Decision Tree model
fpr_best_dt, tpr_best_dt, _ = roc_curve(y_test, y_pred_proba_best_dt)
roc_auc_best_dt = auc(fpr_best_dt, tpr_best_dt)

plt.figure(figsize=(8, 6))
plt.plot(fpr_best_dt, tpr_best_dt, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc_best_dt))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Best Decision Tree)')
plt.legend(loc="lower right")
plt.show()

# Evaluate the best Decision Tree model
accuracy_best_dt = accuracy_score(y_test, y_pred_best_dt)
classification_report_best_dt = classification_report(y_test, y_pred_best_dt)
conf_matrix_best_dt = confusion_matrix(y_test, y_pred_best_dt)

# Display evaluation metrics for the best Decision Tree model
print("\nAccuracy (Best Decision Tree):", accuracy_best_dt)
print("\nClassification Report (Best Decision Tree):")
print(classification_report_best_dt)
print("\nConfusion Matrix (Best Decision Tree):")
print(conf_matrix_best_dt)


# ## threshold adjustment

#%%[markdown]


import numpy as np
from sklearn.metrics import precision_recall_curve


# Find the threshold that maximizes F1-score
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba_best_dt)
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_threshold = thresholds[np.argmax(f1_scores)]

# Adjust predictions based on the optimal threshold
y_pred_threshold_adjusted_dt = (y_pred_proba_best_dt >= optimal_threshold).astype(int)

# Evaluate the model with the adjusted threshold
accuracy_threshold_adjusted_dt = accuracy_score(y_test, y_pred_threshold_adjusted_dt)
classification_report_threshold_adjusted_dt = classification_report(y_test, y_pred_threshold_adjusted_dt)
conf_matrix_threshold_adjusted_dt = confusion_matrix(y_test, y_pred_threshold_adjusted_dt)

# Display evaluation metrics with threshold adjustment
print("\nAccuracy (Best Decision Tree, Threshold Adjusted):", accuracy_threshold_adjusted_dt)
print("\nClassification Report (Best Decision Tree, Threshold Adjusted):")
print(classification_report_threshold_adjusted_dt)
print("\nConfusion Matrix (Best Decision Tree, Threshold Adjusted):")
print(conf_matrix_threshold_adjusted_dt)

# Calculate AUC score after threshold adjustment
roc_auc_threshold_adjusted_dt = roc_auc_score(y_test, y_pred_threshold_adjusted_dt)
print("\nAUC Score (Best Decision Tree, Threshold Adjusted):", roc_auc_threshold_adjusted_dt)

# Plot ROC Curve after threshold adjustment
fpr_threshold_adjusted_dt, tpr_threshold_adjusted_dt, _ = roc_curve(y_test, y_pred_threshold_adjusted_dt)
roc_auc_threshold_adjusted_dt = auc(fpr_threshold_adjusted_dt, tpr_threshold_adjusted_dt)

plt.figure(figsize=(8, 6))
plt.plot(fpr_threshold_adjusted_dt, tpr_threshold_adjusted_dt, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc_threshold_adjusted_dt))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Best Decision Tree, Threshold Adjusted)')
plt.legend(loc="lower right")
plt.show()


# ## XGBoost

# ### Initial model

# ## XGBoost

#%%[markdown]


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Initialize XGBoost model
xgb_model = XGBClassifier(random_state=42)

# Train the XGBoost model on the training set
xgb_model.fit(X_train, y_train)

# Predictions on the testing set
y_pred_xgb = xgb_model.predict(X_test)

# Calculate AUC score for XGBoost model
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
auc_score_xgb = roc_auc_score(y_test, y_pred_proba_xgb)

# Evaluate the XGBoost model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
classification_report_result_xgb = classification_report(y_test, y_pred_xgb)
conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)

# Display evaluation metrics for XGBoost model
print("Accuracy with XGBoost:", accuracy_xgb)
print("\nAUC Score with XGBoost:", auc_score_xgb)
print("\nClassification Report with XGBoost:")
print(classification_report_result_xgb)
print("\nConfusion Matrix with XGBoost:")
print(conf_matrix_xgb)


# ## XGBoost with SMOTE
# 

#%%[markdown]



from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

# Apply SMOTE to the training set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Initialize XGBoost model without hyperparameter tuning
xgb_model_smote = xgb.XGBClassifier(random_state=42)

# Fit the model on the SMOTE-oversampled training set
xgb_model_smote.fit(X_train_smote, y_train_smote)

# Predictions on the testing set
y_pred_xgb_smote = xgb_model_smote.predict(X_test)
y_pred_proba_xgb_smote = xgb_model_smote.predict_proba(X_test)[:, 1]

# Calculate AUC score for the XGBoost model with SMOTE
auc_score_xgb_smote = roc_auc_score(y_test, y_pred_proba_xgb_smote)

# Display evaluation metrics for the XGBoost model with SMOTE
print("Accuracy with XGBoost and SMOTE:", accuracy_score(y_test, y_pred_xgb_smote))
print("AUC Score with XGBoost and SMOTE:", auc_score_xgb_smote)
print("Classification Report with XGBoost and SMOTE:")
print(classification_report(y_test, y_pred_xgb_smote))
print("Confusion Matrix with XGBoost and SMOTE:")
print(confusion_matrix(y_test, y_pred_xgb_smote))



# ### Gave worse results. So we continue WITHOUT doing SMOTE oversampling for XGBoost

# ## Hyperparameter tuning of XGBoost.
# 

#%%[markdown]


from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix

# Define the parameter grid to search
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

# Initialize XGBoost model
xgb_model = xgb.XGBClassifier(random_state=42)

# Create GridSearchCV object
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=-1)

# Perform grid search on the training set
grid_search.fit(X_train, y_train)

# Get the best hyperparameters from the grid search
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Use the best model from grid search
best_xgb_model = grid_search.best_estimator_

# Predictions on the testing set
y_pred_best_xgb = best_xgb_model.predict(X_test)

# Calculate AUC score for the best XGBoost model
y_pred_proba_best_xgb = best_xgb_model.predict_proba(X_test)[:, 1]
auc_score_best_xgb = roc_auc_score(y_test, y_pred_proba_best_xgb)

# Evaluate the best XGBoost model
accuracy_best_xgb = accuracy_score(y_test, y_pred_best_xgb)
classification_report_result_best_xgb = classification_report(y_test, y_pred_best_xgb)
conf_matrix_best_xgb = confusion_matrix(y_test, y_pred_best_xgb)

# Display evaluation metrics for the best XGBoost model
print("\nAccuracy with Best XGBoost Model:", accuracy_best_xgb)
print("AUC Score with Best XGBoost Model:", auc_score_best_xgb)
print("Classification Report with Best XGBoost Model:")
print(classification_report_result_best_xgb)
print("Confusion Matrix with Best XGBoost Model:")
print(conf_matrix_best_xgb)


#%%[markdown]


import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix

# Initialize XGBoost model
xgb_model = xgb.XGBClassifier(random_state=42)

# Train the model on the training set
xgb_model.fit(X_train, y_train)

# Predictions on the testing set
y_pred_xgb = xgb_model.predict(X_test)

# Calculate AUC score for XGBoost model
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
auc_score_xgb = roc_auc_score(y_test, y_pred_proba_xgb)

# Extract feature importances
feature_importance = xgb_model.feature_importances_

# Display the top N most important features
top_n = 10
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
top_features = feature_importance_df.head(top_n)

# Print the top features
print("Top Features:")
print(top_features)

# Evaluate the XGBoost model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
classification_report_result_xgb = classification_report(y_test, y_pred_xgb)
conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)

# Display evaluation metrics for XGBoost model
print("\nAccuracy with XGBoost:", accuracy_xgb)
print("\nAUC Score with XGBoost:", auc_score_xgb)
print("\nClassification Report with XGBoost:")
print(classification_report_result_xgb)
print("\nConfusion Matrix with XGBoost:")
print(conf_matrix_xgb)


# ## Trying out different thresholds and finding the best balance at 0.1

#%%[markdown]


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


# Adjust the threshold
new_threshold_more = 0.1
y_pred_proba_best_xgb_adj_more = (best_xgb_model.predict_proba(X_test)[:, 1] >= new_threshold_more).astype(int)

# Evaluate the adjusted XGBoost model with the new threshold
accuracy_best_xgb_adj_more = accuracy_score(y_test, y_pred_proba_best_xgb_adj_more)
classification_report_result_best_xgb_adj_more = classification_report(y_test, y_pred_proba_best_xgb_adj_more)
conf_matrix_best_xgb_adj_more = confusion_matrix(y_test, y_pred_proba_best_xgb_adj_more)
auc_score_best_xgb_adj_more = roc_auc_score(y_test, y_pred_proba_best_xgb_adj_more)

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_best_xgb_adj_more)
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, label=f'AUC = {auc_score_best_xgb_adj_more:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Display evaluation metrics for the adjusted XGBoost model with the new threshold
print("\nAccuracy of XGB with Adjusted Threshold", accuracy_best_xgb_adj_more)
print("AUC Score of XGB with Adjusted Threshold:", auc_score_best_xgb_adj_more)
print("Classification Report of XGB with Adjusted Threshold:")
print(classification_report_result_best_xgb_adj_more)
print("Confusion Matrix of XGB with Adjusted Threshold:")
print(conf_matrix_best_xgb_adj_more)


# ## Feature Importance

#%%[markdown]


# To find out the top 10 features in predicting the target variable.
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix

# Initialize XGBoost model
xgb_model = xgb.XGBClassifier(random_state=42)

# Train the model on the training set
xgb_model.fit(X_train, y_train)

# Predictions on the testing set
y_pred_xgb = xgb_model.predict(X_test)

# Calculate AUC score for XGBoost model
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
auc_score_xgb = roc_auc_score(y_test, y_pred_proba_xgb)

# Extract feature importances
feature_importance = xgb_model.feature_importances_

# Display the top N most important features
top_n = 10
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
top_features = feature_importance_df.head(top_n)

# Print the top features
print("Top Features:")
print(top_features)

# Plot the top features
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
plt.title('Top 10 Important Features in XGBoost Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()































