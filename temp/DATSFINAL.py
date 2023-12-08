#%%
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
warnings.filterwarnings('ignore')
%matplotlib inline



from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from imblearn.over_sampling import RandomOverSampler


#%%
df=pd.read_csv("fraud_oracle.csv.csv")
#%%
#The first 5 Rows in the data set is shown below
df.head()
#%%
#The last 5 Rows in the data set is shown below
df.tail()
#%%
#[x,y]=[rows,columns]
df.shape
#%%
#The column names in the data set is shown below
df.columns
for column in df.columns:
    print(column)
#%%
#The Datatypes of variables in the data set is shown below
df.dtypes
#%%
#Checking for the null values in the dataframe
df.isnull()
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values in the Dataset')
plt.xticks(rotation=90)
plt.show()
# %%
#description of the dataset
df.describe()
# %%
#%%
#Dataframe Info
df.info()
#%%
print('Out of all of records not commited fraud is: ',round(len(df[df['FraudFound_P']==0])/len(df)*100) ,'%')
print('Out of all of records commited fraud is    : ',round(len(df[df['FraudFound_P']==1])/len(df)*100) ,'%')

#%%


























###############types to change according to smart question !!###################


#plots are below ----->
#%%
make_list = ['Lexus', 'Ferrari', 'Porsche', 'Jaguar', 'Mercedes', 'BMW', 'Nissan',
             'Saturn', 'Dodge', 'Mercury', 'Saab', 'VW', 'Ford', 'Acura',
             'Chevrolet', 'Mazda', 'Honda', 'Toyota', 'Pontiac']

# Create a filtered DataFrame based on the provided makes
filtered_df = df[make_list]

# Count the occurrences of each make
make_counts = filtered_df['Make'].value_counts()
# Plotting the bar graph
plt.figure(figsize=(10, 6))
make_counts.plot(kind='bar', color='skyblue')
plt.title('Number of Cars by Make')
plt.xlabel('Make')
plt.ylabel('Number of Cars')
plt.xticks(rotation=45, ha='right')
plt.show()
#%%
Make_list=['Lexus', 'Ferrari', 'Porche', 'Jaguar', 'Mecedes', 'BMW', 'Nisson',
       'Saturn', 'Dodge', 'Mercury', 'Saab', 'VW', 'Ford', 'Accura',
       'Chevrolet', 'Mazda', 'Honda', 'Toyota', 'Pontiac']
plt.bar(Make_list,df1.groupby('Make').count()['FraudFound_P'].sort_values())
plt.xticks(Make_list,rotation='vertical')
plt.show()
df.groupby('Make').count()['FraudFound_P'].sort_values()
#%%
Days_list = ['Sunday', 'Saturday', 'Wednesday', 'Thursday', 'Tuesday', 'Friday', 'Monday']
day_counts = df1.groupby('DayOfWeek').count()['FraudFound_P'].sort_values()
plt.bar(Days_list, day_counts)
for i, count in enumerate(day_counts):
    plt.text(i, count + 0.1, str(count), ha='center', va='bottom')

plt.title('Fraud Cases by Day of Week')
plt.xlabel('Day of Week')
plt.xticks(rotation=45)
plt.ylabel('Number of Fraud Cases')
plt.show()

#%%
fig, ax = plt.subplots(figsize = (20, 5))

ax.hist(df['Age'], bins = 25, edgecolor = 'black', alpha = 0.7, color = 'skyblue', density = True)
df['Age'].plot(kind = 'kde', color = 'red', ax = ax)

ax.set_xlabel('Age')
ax.set_ylabel('Count / Density')
ax.set_title('Age Distribution Histogram with Density Curve')
ax.legend(['Density Curve', 'Histogram'])
plt.show()
# %%
plt.figure(figsize = (20, 6))
ax = df["MaritalStatus"].value_counts().plot(kind = 'bar', color = ["red","blue","green","yellow"], rot = 0)
ax.set_xticklabels(('Single', 'Married','Divorced','Widow'))
plt.title("MaritalStatus")
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + 0.25, p.get_height() + 1), ha = 'center', va = 'bottom', color = 'black')
    ax.tick_params(axis = 'both', labelsize = 15)
plt.xlabel('Employment Type', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20)
plt.title("MaritalStatus vs frequency")
plt.ylabel('Number of Occurrences', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20);
# %%
plt.figure(figsize=(16, 6))
fault_counts = df["Fault"].value_counts()[:10].sort_values(ascending=False)
ax = fault_counts.plot(kind='bar', color=sns.color_palette('inferno'), edgecolor='black')
for i, v in enumerate(fault_counts):
    ax.text(i, v + 0.1, str(v), fontweight='bold', ha='center')
plt.xlabel('Fault', weight="bold", color="#D71313", fontsize=14, labelpad=20)
plt.ylabel('\nNumber of Occurrences', weight="bold", color="#D71313", fontsize=14, labelpad=20)
plt.xticks(rotation=0, ha='center', fontsize=16)
plt.title("Fault vs frequency")
plt.tight_layout()
plt.show()
# %%
plt.figure(figsize = (20, 6))
ax = df["PolicyType"].value_counts().plot(kind = 'bar', color = ["red","blue","green","yellow"], rot = 0)
plt.title("PolicyType vs frequency")
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + 0.25, p.get_height() + 1), ha = 'center', va = 'bottom', color = 'black')
    ax.tick_params(axis = 'both', labelsize = 15)
plt.xlabel('Size', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20)
plt.ylabel('Number of Occurrences', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20);

# %%
plt.figure(figsize=(16, 6))
value_counts = df["VehicleCategory"].value_counts()[:10].sort_values(ascending=True)
ax = value_counts.plot(kind='barh', color=sns.color_palette('tab20'), edgecolor='black')
for index, value in enumerate(value_counts):
    ax.text(value, index, str(value), ha='center', va='center', fontsize=12, color='black')

plt.xlabel('Number of Occurrences', weight="bold", color="#D71313", fontsize=14, labelpad=20)
plt.ylabel('Vehicle Category', weight="bold", color="#D71313", fontsize=14, labelpad=20)
plt.title("VehicleCategory vs Frequency", fontsize=16)
plt.xticks(rotation=0, ha='center', fontsize=12)
plt.tight_layout()
plt.show()
# %%
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.xticks(rotation=45, ha='center', fontsize=12)
plt.show()
# %%
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='FraudFound_P', data=df, palette='viridis')
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.title('Histogram of Fraud Reported')
plt.xlabel('Fraud Reported')
plt.ylabel('Count')
plt.show()
# %%
plt.figure(figsize=(12, 6))
ax = sns.histplot(x='Age', data=df, bins=20, kde=False, color=(0.2, 0.7, 0.3))
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
#%%
####################################smart questions##################################
# %%
plt.figure(figsize=(15,8))
sns.countplot(data=df, x="VehiclePrice", hue="FraudFound_P")
# %%
contingency_table = pd.crosstab(df['VehiclePrice'], df['FraudFound_P'])

# Perform chi-square test of independence
chi2, p_value, _, expected = stats.chi2_contingency(contingency_table)

print("Chi-square statistic:", chi2)
print("p-value:", p_value)

'''
Null Hypothesis (H0): There is no significant association between the price of vehicles and the likelihood of fraud. The proportions of fraudulent claims are the same across different price categories of vehicles.

Alternative Hypothesis (HA): There is a significant association between the price of vehicles and the likelihood of fraud. The proportions of fraudulent claims differ across different price categories of vehicles.

'''
# %%
contingency_table = pd.crosstab(df['Fault'], df['FraudFound_P'])

# Perform chi-square test of independence
chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)

print("Chi-square statistic:", chi2)
print("p-value:", p_value)
sns.countplot(data=df, x="Fault", hue="FraudFound_P")


'''
Null Hypothesis (H0): There is no significant association between the type of fault (at-fault or not-at-fault) and the occurrence of fraud. The proportions of fraudulent claims are the same for both types of fault.

Alternative Hypothesis (HA): There is a significant association between the type of fault and the occurrence of fraud. The proportions of fraudulent claims differ between at-fault and not-at-fault claims.

'''
# %%
contingency_table = pd.crosstab(df['PoliceReportFiled'], df['FraudFound_P'])

# Perform chi-square test of independence
chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)

print("Chi-square statistic:", chi2)
print("p-value:", p_value)


sns.countplot(data=df, x="PoliceReportFiled", hue="FraudFound_P")

'''
Null Hypothesis (H0): There is no significant association between the presence of a police report and the occurrence of fraud. The proportions of fraudulent claims are the same for claims with and without a police report.

Alternative Hypothesis (HA): There is a significant association between the presence of a police report and the occurrence of fraud. The proportions of fraudulent claims differ between claims with and without a police report.
'''
# %%
# Create a contingency table
contingency_table = pd.crosstab(df['WitnessPresent'], df['FraudFound_P'])

# Perform chi-square test of independence
chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)

print("Chi-square statistic:", chi2)
print("p-value:", p_value)


sns.countplot(data=df, x="WitnessPresent", hue="FraudFound_P")


''''
Null Hypothesis (H0): There is no significant association between the presence of a witness and the occurrence of fraud. The proportions of fraudulent claims are the same for claims with and without a witness.

Alternative Hypothesis (HA): There is a significant association between the presence of a witness and the occurrence of fraud. The proportions of fraudulent claims differ between claims with and without a witness.
'''


# %%
#########################models##################################


# Creating a list of categorical features
cat_features = ["Month", "DayOfWeek", "Make", "AccidentArea", "DayOfWeekClaimed", "MonthClaimed", "Sex", "MaritalStatus", "Fault", "PolicyType", "VehicleCategory", "PoliceReportFiled", "WitnessPresent", "AgentType", "AddressChange_Claim", "BasePolicy"]

# One-hot encoding the categorical features
ddf = pd.get_dummies(df, columns=cat_features)

# Separating the features and the target variable
X = ddf.drop("FraudFound_P", axis=1)
y = ddf["FraudFound_P"]

# Creating a list of classifiers
classifiers = [LogisticRegression(max_iter=1000), SVC(max_iter=1000), DecisionTreeClassifier(), RandomForestClassifier(), KNeighborsClassifier(), GaussianNB(), GradientBoostingClassifier()]

# Creating a stratified k-fold object with 5 splits
skf = StratifiedKFold(n_splits=5)

# Looping over the classifiers
for clf in classifiers:
    # Initializing empty lists to store the evaluation metrics for each fold
    accuracies = []
    precisions = []
    recalls = []

    # Looping over the train and test indices of each fold
    for train_index, test_index in skf.split(X, y):
        # Splitting the data into train and test sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # Fitting the classifier on the train set
        clf.fit(X_train, y_train)
        # Predicting on the test set
        y_pred = clf.predict(X_test)
        # Calculating the evaluation metrics and appending them to the respective lists
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)

    # Printing the mean and standard deviation of the evaluation metrics for each classifier
    print(f"{clf.__class__.__name__}: Accuracy={np.mean(accuracies):.3f} (+/- {np.std(accuracies):.3f}), Precision={np.mean(precisions):.3f} (+/- {np.std(precisions):.3f}), Recall={np.mean(recalls):.3f} (+/- {np.std(recalls):.3f})")
# %%
# Creating a list of categorical features
cat_features = ["Month", "DayOfWeek", "Make", "AccidentArea", "DayOfWeekClaimed", "MonthClaimed", "Sex", "MaritalStatus", "Fault", "PolicyType", "VehicleCategory", "PoliceReportFiled", "WitnessPresent", "AgentType", "AddressChange_Claim", "BasePolicy"]

# One-hot encoding the categorical features
ddf = pd.get_dummies(df, columns=cat_features)

# Separating the features and the target variable
X = ddf.drop("FraudFound_P", axis=1)
y = ddf["FraudFound_P"]

# Creating a RandomOverSampler object
oversampler = RandomOverSampler()

# Applying oversampling to the dataset
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Creating a stratified k-fold object with 5 splits for outer loop
skf_outer = StratifiedKFold(n_splits=5)

# Looping over the classifiers
for clf in [RandomForestClassifier()]:
    # Initializing empty lists to store the evaluation metrics for each fold
    accuracies = []
    precisions = []
    recalls = []

    # Creating a stratified k-fold object with 5 splits for inner loop
    skf_inner = StratifiedKFold(n_splits=5)

    # Defining the parameter grid for the classifier
    param_grid = {
        "RandomForestClassifier": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10]
        }
    }

    # Performing nested cross-validation for parameter tuning
    grid_search = GridSearchCV(clf, param_grid[clf.__class__.__name__], cv=skf_inner, scoring="accuracy")
    for train_index_outer, test_index_outer in skf_outer.split(X_resampled, y_resampled):
        # Splitting the resampled data into outer train and test sets
        X_train_outer, X_test_outer = X_resampled.iloc[train_index_outer], X_resampled.iloc[test_index_outer]
        y_train_outer, y_test_outer = y_resampled.iloc[train_index_outer], y_resampled.iloc[test_index_outer]

        # Fitting the grid search on the outer train set
        grid_search.fit(X_train_outer, y_train_outer)

        # Predicting on the outer test set using the best estimator
        y_pred_outer = grid_search.best_estimator_.predict(X_test_outer)

        # Calculating the evaluation metrics and appending them to the respective lists
        accuracy_outer = accuracy_score(y_test_outer, y_pred_outer)
        precision_outer = precision_score(y_test_outer, y_pred_outer, zero_division=0)
        recall_outer = recall_score(y_test_outer, y_pred_outer, zero_division=0)
        accuracies.append(accuracy_outer)
        precisions.append(precision_outer)
        recalls.append(recall_outer)

    # Printing the mean and standard deviation of the evaluation metrics for each classifier
    print(f"{clf.__class__.__name__}: Accuracy={np.mean(accuracies):.3f} (+/- {np.std(accuracies):.3f}), Precision={np.mean(precisions):.3f} (+/- {np.std(precisions):.3f}), Recall={np.mean(recalls):.3f} (+/- {np.std(recalls):.3f})")
# %%
