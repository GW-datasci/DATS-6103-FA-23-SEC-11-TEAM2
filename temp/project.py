"""
dataset link= https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection/
 target variable = policy


smart questions

What is the distribution of deductible amounts chosen by policyholders?
How does the age of the vehicle affect the type of coverage selected?
Is there a link between the number of previous claims and the sort of insurance you have?
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline



df=pd.read_csv('/kaggle/input/vehicle-claim-fraud-detection/fraud_oracle.csv')
df.head()