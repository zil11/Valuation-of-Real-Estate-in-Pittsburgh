# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 14:40:41 2023

@author: jeffe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'C:\Users\jeffe\Desktop\PITT\ECON 2814 Evidence-Based Analysis\Assignments\Assignment 3 - King Estate\2023 Property Assessments Parcel Data.csv')


# Filter data

df = df[df['PROPERTYCITY'] == 'PITTSBURGH'] 
df = df[df['PROPERTYZIP'] == 15206]
df = df[df['CLASS'] == 'R'] 
#df = df[df['SALEDESC'] == 'VALID SALE']
df = df[df['LOTAREA'] > 0]
df = df[df['CLEANGREEN'] != 'Y']
#df = df[df['MUNICODE'] <= 114]
df = df[df['SALEDATE'] != '05-16-1867']
df = df[df['LOTAREA'] < 120000]
df = df[df['USEDESC'] == 'SINGLE FAMILY']

# Check NA

df.columns[df.isnull().any()].tolist()

# Decode sale date

df['SALEDATE'] = pd.to_datetime(df['SALEDATE'], format='%m/%d/%Y')

# Add a new column for the month of the year

df['MONTH'] = df['SALEDATE'].dt.month

df['YEAR'] = df['SALEDATE'].dt.year

df['DATE'] = df['SALEDATE'].dt.date

# Drop NAs

df = df.dropna(subset=["TOTALROOMS", 'LOTAREA', 'STORIES', 'FIREPLACES','STORIES','BEDROOMS','FULLBATHS','FIREPLACES','BSMTGARAGE','FINISHEDLIVINGAREA','YEAR', 'HALFBATHS'])

# Summary Statistics

stat = df.describe()

# Data Manipulation

df.loc[df['FIREPLACES'] > 0, 'FIREPLACES'] = 1     
df.loc[df['BSMTGARAGE'] > 0, 'BSMTGARAGE'] = 1   
df.loc[df['GRADEDESC'] == 'BELOW AVERAGE -', 'GRADEDESC'] = -5
df.loc[df['GRADEDESC'] == 'BELOW AVERAGE', 'GRADEDESC'] = -4
df.loc[df['GRADEDESC'] == 'BELOW AVERAGE +', 'GRADEDESC'] = -3
df.loc[df['GRADEDESC'] == 'AVERAGE -', 'GRADEDESC'] = -2
df.loc[df['GRADEDESC'] == 'AVERAGE', 'GRADEDESC'] = -1
df.loc[df['GRADEDESC'] == 'AVERAGE +', 'GRADEDESC'] = 1
df.loc[df['GRADEDESC'] == 'GOOD -', 'GRADEDESC'] = 2
df.loc[df['GRADEDESC'] == 'GOOD', 'GRADEDESC'] = 3
df.loc[df['GRADEDESC'] == 'GOOD +', 'GRADEDESC'] = 4
df.loc[df['GRADEDESC'] == 'VERY GOOD -', 'GRADEDESC'] = 5
df.loc[df['GRADEDESC'] == 'VERY GOOD', 'GRADEDESC'] = 6
df.loc[df['GRADEDESC'] == 'VERY GOOD +', 'GRADEDESC'] = 7
df.loc[df['GRADEDESC'] == 'EXCELLENT -', 'GRADEDESC'] = 8
df.loc[df['GRADEDESC'] == 'EXCELLENT', 'GRADEDESC'] = 9
df.loc[df['GRADEDESC'] == 'EXCELLENT +', 'GRADEDESC'] = 10
df = df[df['GRADEDESC'] != 'HIGHEST COST -']
df = df[df['GRADEDESC'] != 'Highest Cost']
df.loc[df['CDU'] == 'VERY POOR', 'CDU'] = -10
df.loc[df['CDU'] == 'POOR', 'CDU'] = -6
df.loc[df['CDU'] == 'FAIR', 'CDU'] = -3
df.loc[df['CDU'] == 'AVERAGE', 'CDU'] = -1
df.loc[df['CDU'] == 'GOOD', 'CDU'] = 3
df.loc[df['CDU'] == 'VERY GOOD', 'CDU'] = 6
df = df[df['CDU'] != 'UNSOUND']
df = df[df['CDU'] != 'EXCELLENT']
df['AGE'] = 2023 - df['YEARBLT']

# Remove COVID period

df = df[(df['SALEDATE'].dt.year < 2020)]

# Visualization of Distributions AND Correlation Matrix

plt.figure(figsize=(20,20))
corr = sns.heatmap(df.corr(),annot = True,fmt = ".2f",cbar = True)
plt.xticks(rotation=90)
plt.yticks(rotation = 0)
plt.show()

plt.scatter(x = 'GRADEDESC', y = 'FAIRMARKETTOTAL', data = df)
plt.show()

plt.scatter(x = 'CDU', y = 'FAIRMARKETTOTAL', data = df)
plt.show()

plt.scatter(x = 'AGE', y = 'FAIRMARKETBUILDING', data = df)
plt.show()

sns.histplot(df, x = 'FAIRMARKETTOTAL', bins = 50, kde = True, color = 'k')
plt.title('Fair Market Value Total Distribution (ZIP: 15206)')
plt.xlabel('Fair Market Value Total')
plt.ylabel('Frequency')
plt.show()

df['logfmt'] = np.log(df['FAIRMARKETTOTAL']) 
sns.histplot(df, x = 'logfmt', bins = 50, kde = True, color = 'r')
plt.title('Log(Fair Market Value Total) Distribution (ZIP: 15206)')
plt.xlabel('Log(Fair Market Value Total)')
plt.ylabel('Frequency')
plt.show()

sns.histplot(df, x = 'LOTAREA', bins = 50, kde = True, color = 'k')
plt.title('Lot Area Distribution (ZIP: 15206)')
plt.xlabel('Lot Area')
plt.ylabel('Frequency')
plt.show()

df['loglot'] = np.log(df['LOTAREA']) 
sns.histplot(df, x = 'loglot', bins = 50, kde = True, color = 'r')
plt.title('Log(Lot Area) Distribution (ZIP: 15206)')
plt.xlabel('Log(Lot Area)')
plt.ylabel('Frequency')
plt.show()

sns.histplot(df, x = 'FINISHEDLIVINGAREA', bins = 50, kde = True, color = 'k')
plt.title('Finished Living Area Distribution (ZIP: 15206)')
plt.xlabel('Finished Living Area')
plt.ylabel('Frequency')
plt.show()

df['logliving'] = np.log(df['FINISHEDLIVINGAREA'])
sns.histplot(df, x = 'logliving', bins = 50, kde = True, color = 'r')
plt.title('Log(Finished Living Area) Distribution (ZIP: 15206)')
plt.xlabel('Log(Finished Living Area)')
plt.ylabel('Frequency')
plt.show()

sns.histplot(df, x = 'AGE', bins = 50, kde = True, color = 'r')
plt.title('Building Age Distribution (ZIP: 15206)')
plt.xlabel('Building Age Distribution)')
plt.ylabel('Frequency')
plt.show()
df = df[df['AGE'] <= 150]
df = df[df['AGE'] >= 50]

# Split into test and train
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
label = df['logfmt']
features = df[['loglot','BEDROOMS','FULLBATHS','HALFBATHS','FIREPLACES','BSMTGARAGE','logliving','AGE']]
scaler = StandardScaler()
#features = scaler.fit_transform(features)
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size = 0.2, random_state=(2))

# Method 1: Regression
import statsmodels.api as sm
x_train = sm.add_constant(x_train)
lm = sm.OLS(y_train, x_train).fit()
predictions = lm.predict(x_test) 
lm_sum = lm.summary()
print(lm_sum)

# Method 2: Gradient Boosting Regression 

df = pd.read_csv(r'C:\Users\jeffe\Desktop\PITT\ECON 2814 Evidence-Based Analysis\Assignments\Assignment 3 - King Estate\2023 Property Assessments Parcel Data.csv')


# Filter data

df = df[df['PROPERTYCITY'] == 'PITTSBURGH'] 
#df = df[df['PROPERTYZIP'] == 15206]
df = df[df['CLASS'] == 'R'] 
df = df[df['SALEDESC'] == 'VALID SALE']
df = df[df['USECODE'] == 10]
df = df[df['CLEANGREEN'] != 'Y']
#df = df[df['MUNICODE'] <= 114]
df = df[df['SALEDATE'] != '05-16-1867']
df = df[df['LOTAREA'] < 200000]
df = df[df['LOTAREA'] > 10000]
df = df[df['FAIRMARKETTOTAL'] >= 100000]
df = df[df['FAIRMARKETTOTAL'] <= 1500000]

# Decode sale date

df['SALEDATE'] = pd.to_datetime(df['SALEDATE'], format='%m/%d/%Y')
df['MONTH'] = df['SALEDATE'].dt.month
df['YEAR'] = df['SALEDATE'].dt.year
df['DATE'] = df['SALEDATE'].dt.date

# Drop NAs
df = df.dropna(subset=["TOTALROOMS", 'LOTAREA', 'STORIES', 'FIREPLACES','STORIES','BEDROOMS','FULLBATHS','HALFBATHS','FIREPLACES','BSMTGARAGE','FINISHEDLIVINGAREA','YEAR', 'CDU', 'GRADEDESC'])

# Data Manipulation

df.loc[df['FIREPLACES'] > 0, 'FIREPLACES'] = 1     
df.loc[df['BSMTGARAGE'] > 0, 'BSMTGARAGE'] = 1   
df.loc[df['GRADEDESC'] == 'BELOW AVERAGE -', 'GRADEDESC'] = -5
df.loc[df['GRADEDESC'] == 'BELOW AVERAGE', 'GRADEDESC'] = -4
df.loc[df['GRADEDESC'] == 'BELOW AVERAGE +', 'GRADEDESC'] = -3
df.loc[df['GRADEDESC'] == 'AVERAGE -', 'GRADEDESC'] = -2
df.loc[df['GRADEDESC'] == 'AVERAGE', 'GRADEDESC'] = -1
df.loc[df['GRADEDESC'] == 'AVERAGE +', 'GRADEDESC'] = 1
df.loc[df['GRADEDESC'] == 'GOOD -', 'GRADEDESC'] = 2
df.loc[df['GRADEDESC'] == 'GOOD', 'GRADEDESC'] = 3
df.loc[df['GRADEDESC'] == 'GOOD +', 'GRADEDESC'] = 4
df.loc[df['GRADEDESC'] == 'VERY GOOD -', 'GRADEDESC'] = 5
df.loc[df['GRADEDESC'] == 'VERY GOOD', 'GRADEDESC'] = 6
df.loc[df['GRADEDESC'] == 'VERY GOOD +', 'GRADEDESC'] = 7
df.loc[df['GRADEDESC'] == 'EXCELLENT -', 'GRADEDESC'] = 8
df.loc[df['GRADEDESC'] == 'EXCELLENT', 'GRADEDESC'] = 9
df.loc[df['GRADEDESC'] == 'EXCELLENT +', 'GRADEDESC'] = 10
df = df[df['GRADEDESC'] != 'HIGHEST COST -']
df = df[df['GRADEDESC'] != 'Highest Cost']
df.loc[df['CDU'] == 'VERY POOR', 'CDU'] = -10
df.loc[df['CDU'] == 'POOR', 'CDU'] = -6
df.loc[df['CDU'] == 'FAIR', 'CDU'] = -3
df.loc[df['CDU'] == 'AVERAGE', 'CDU'] = -1
df.loc[df['CDU'] == 'GOOD', 'CDU'] = 3
df.loc[df['CDU'] == 'VERY GOOD', 'CDU'] = 6
df = df[df['CDU'] != 'UNSOUND']
df = df[df['CDU'] != 'EXCELLENT']
df['AGE'] = 2022 - df['YEARBLT']
df = df[df['AGE'] != 0]


from sklearn.metrics import explained_variance_score as evs # evaluation metric
from sklearn.metrics import r2_score as r2 # evaluation metric
from sklearn.ensemble import GradientBoostingRegressor

label = df['FAIRMARKETTOTAL']
features = df[['LOTAREA','BEDROOMS','FULLBATHS','HALFBATHS','FIREPLACES','BSMTGARAGE','FINISHEDLIVINGAREA','AGE']]
features = scaler.fit_transform(features)
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size = 0.2, random_state=(2))

gbr = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01, max_depth=4, random_state=42)
gbr.fit(x_train, y_train)

from sklearn import ensemble
clf = ensemble.GradientBoostingRegressor(n_estimators=1000, max_depth=4, min_samples_split=5, learning_rate=0.01, loss='ls')
clf.fit(x_train, y_train)
clf.score(x_test, y_test)
y_pred = clf.predict(x_test)
print("Model Accuracy: %.3f" % clf.score(x_test, y_test))
r2(y_test, y_pred)
king = np.array([80368, 8, 2, 2, 4, 9286, 143], dtype=np.float64)
king = king.reshape(1, -1)
king = scaler.fit_transform(king)
king_pred = clf.predict(king)

