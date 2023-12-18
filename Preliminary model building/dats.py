#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
# %%
df=pd.read_csv("fraud_oracle.csv.csv")
df.head()
# %%
df.columns
# %%
df.info()
# %%
df.describe()
# %%
print('Out of all % of records not commited fraud==> ',len(df[df['FraudFound_P']==0])/len(df)*100)
print('Out of all % of records commited fraud==> ',len(df[df['FraudFound_P']==1])/len(df)*100)
# %%
from sklearn.utils import resample

oversamp=resample(df[df['FraudFound_P']==1],
                 replace=True,
                 n_samples=len(df[df['FraudFound_P']==0]),
                 random_state=30)
df1=pd.concat([oversamp, df[df['FraudFound_P']==0]])
# %%
df1.shape

# %%
print('Out of all % of records not commited fraud==> ',len(df1[df1['FraudFound_P']==0])/len(df1)*100)
print('Out of all % of records commited fraud==> ',len(df1[df1['FraudFound_P']==1])/len(df1)*100)
# After Oversampling of commited Fraud records
# %%
df1.head()

# %%
Months_list=['Nov','Jul','Dec','Oct','Sep','Aug','Apr','Jun','Feb','Jan','May','Mar']
plt.bar(Months_list,df1.groupby('Month').count()['FraudFound_P'].sort_values())
plt.show()
# %%
Days_list=[ 'Sunday', 'Saturday','Wednesday','Thursday','Tuesday', 'Friday', 'Monday']
plt.bar(Days_list,df1.groupby('DayOfWeek').count()['FraudFound_P'].sort_values())
plt.show()
# %%
pd.DataFrame(df1.groupby('Make').count()['FraudFound_P'].sort_values()).index
# %%
Make_list=['Lexus', 'Ferrari', 'Porche', 'Jaguar', 'Mecedes', 'BMW', 'Nisson',
       'Saturn', 'Dodge', 'Mercury', 'Saab', 'VW', 'Ford', 'Accura',
       'Chevrolet', 'Mazda', 'Honda', 'Toyota', 'Pontiac']
plt.bar(Make_list,df1.groupby('Make').count()['FraudFound_P'].sort_values())
plt.xticks(Make_list,rotation='vertical')
plt.show()
# %%
df1.groupby('Make').count()['FraudFound_P'].sort_values()
# %%
plt.pie(df1.groupby('Sex').count()['FraudFound_P'].sort_values(),labels=['Female','Male'],explode=[0.1,0],shadow=True,autopct='%.2f')
plt.show()
# %%
plt.bar(['Rural','Urban'],df1.groupby('AccidentArea').count()['FraudFound_P'].sort_values())
plt.show()
# %%
df1.head()

# %%
df1.columns

# %%
df2=df1[['Make', 'AccidentArea','Sex','MaritalStatus', 'Age', 'Fault', 'PolicyType', 'VehicleCategory',
         'VehiclePrice','Deductible', 'DriverRating','PastNumberOfClaims', 'AgeOfVehicle',
         'AgeOfPolicyHolder', 'PoliceReportFiled', 'WitnessPresent', 'AgentType','NumberOfSuppliments', 
         'AddressChange_Claim', 'NumberOfCars', 'Year','BasePolicy','FraudFound_P']]
# %%
df2.head()

# %%
df2['Make']=df2['Make'].replace({'Lexus':0, 'Ferrari':1, 'Porche':2, 'Jaguar':3, 'Mecedes':4, 'BMW':5, 'Nisson':6,
       'Saturn':7, 'Dodge':8, 'Mercury':9, 'Saab':10, 'VW':11, 'Ford':12, 'Accura':13, 
       'Chevrolet':14, 'Mazda':15, 'Honda':16, 'Toyota':17, 'Pontiac':18})
# %%
from sklearn.preprocessing import LabelEncoder
df2['Fault']=LabelEncoder().fit_transform(df2['Fault'])
# %%
df2.AccidentArea=LabelEncoder().fit_transform(df2['AccidentArea'])
df2.Sex=LabelEncoder().fit_transform(df2['Sex'])
df2.AccidentArea=LabelEncoder().fit_transform(df2['AccidentArea'])
df2.MaritalStatus=LabelEncoder().fit_transform(df2['MaritalStatus'])
df2.PolicyType=LabelEncoder().fit_transform(df2['PolicyType'])
df2.VehicleCategory=LabelEncoder().fit_transform(df2['VehicleCategory'])
df2.PoliceReportFiled=LabelEncoder().fit_transform(df2['PoliceReportFiled'])
df2.WitnessPresent=LabelEncoder().fit_transform(df2['WitnessPresent'])
df2.AgentType=LabelEncoder().fit_transform(df2['AgentType'])
df2.BasePolicy=LabelEncoder().fit_transform(df2['BasePolicy'])
# %%
df2.head()

# %%
df2.NumberOfSuppliments=df2['NumberOfSuppliments'].replace('none',0)
df2.NumberOfSuppliments=df2['NumberOfSuppliments'].str.replace('\D','',regex=True)
df2.NumberOfSuppliments=df2.NumberOfSuppliments.fillna(0)
# %%
df2.AddressChange_Claim=df2['AddressChange_Claim'].replace('no change',0)
df2.AddressChange_Claim=df2['AddressChange_Claim'].str.replace('\D','',regex=True)
df2.AddressChange_Claim=df2['AddressChange_Claim'].fillna(0)
# %%
df2.AgeOfVehicle=df2['AgeOfVehicle'].replace('new',1)
df2.AgeOfVehicle=df2['AgeOfVehicle'].str.replace('\D','',regex=True)
df2.AgeOfVehicle=df2['AgeOfVehicle'].fillna(1)
# %%
df2.iloc[:,8:18]

# %%
import re

def strcon(a):
    number=re.findall(r'\d+',a)
    if len(number)==2:
        return (int(number[0])+int(number[1]))/2
    elif len(number)==1:
        return int(number[0])
    else:
        return 0
# %%
df2.VehiclePrice=df2['VehiclePrice'].apply(strcon)
df2.AgeOfPolicyHolder=round(df2['AgeOfPolicyHolder'].apply(strcon))
df2.NumberOfCars=round(df2['NumberOfCars'].apply(strcon))
# %%
df2.PastNumberOfClaims=df2['PastNumberOfClaims'].replace({'1':1, 'none':0, '2 to 4':2, 'more than 4':4})
# %%
df2.AgeOfVehicle=df2['AgeOfVehicle'].astype('int')
df2.NumberOfSuppliments=df2['NumberOfSuppliments'].astype('int')
df2.AddressChange_Claim=df2['AddressChange_Claim'].astype('int')
# %%
df2.info()

# %%
plt.figure(figsize=(18,18))
sns.heatmap(df2.corr(),annot=True)
plt.show()
# %%
df2.columns
# %%
df2.drop(['BasePolicy','Fault','PolicyType','PastNumberOfClaims','Year','NumberOfSuppliments','AgentType',
         'PastNumberOfClaims', 'AgeOfVehicle','AgeOfPolicyHolder', 'PoliceReportFiled', 'WitnessPresent',
         'VehicleCategory','Make','AccidentArea'],axis='columns',inplace=True)
# %%
df2.head()

# %%
x=df2.drop('FraudFound_P',axis='columns')
y=df2.FraudFound_P
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=20,test_size=0.2)
# %%
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train,y_train)
# %%
model.score(x_test,y_test)

# %%
x_test.iloc[:5]

# %%
y_test.iloc[:5]

# %%
model.score(x_train,y_train)


# %%
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(model.predict(x_test),y_test),annot=True)
plt.show()
# %%
confusion_matrix(model.predict(x_test),y_test)

# %%
#################################

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
plt.figure(figsize = (16, 6))
df["Fault"].value_counts()[:10].sort_values(ascending = False).plot(kind = 'bar', color = sns.color_palette('inferno'), edgecolor = 'black')
plt.xlabel('Fault', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20)
plt.ylabel('\nNumber of Occurrences', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20);
plt.xticks(rotation = 0, ha = 'center', fontsize = 16)
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
plt.figure(figsize = (16, 6))
df["VehicleCategory"].value_counts()[:10].sort_values(ascending = True).plot(kind = 'barh', color = sns.color_palette('tab20'), edgecolor = 'black')
plt.xlabel('Color', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20)
plt.ylabel('\nNumber of Occurrences', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20);
plt.xticks(rotation = 0, ha = 'center', fontsize = 16)
plt.title("VehicleCategory vs frequency")
plt.tight_layout()
plt.show()
# %%
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
#%%
# Countplot of policy types
plt.figure(figsize=(10, 6))
sns.countplot(x='PolicyType', data=df)
plt.title('Count of Policies by Type')
plt.xlabel('Policy Type')
plt.xticks(rotation=45)
plt.ylabel('Count')
plt.show()
# %%
