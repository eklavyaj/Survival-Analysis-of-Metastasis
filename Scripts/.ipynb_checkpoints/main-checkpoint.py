####### Libraries #######

import pandas as pd
import numpy as np
from math import fabs
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error    

from cobra_estimator import Cobra
####### Loading and Cleaning Dataset #######

df = pd.read_csv("../Data/selected.csv", index_col = False)
df.drop(["Unnamed: 0"], inplace = True, axis = 1)
df = df.replace(' NA', np.nan)
df = df.replace('NA', np.nan)
df['NPI'] = df['NPI'].astype('float')
df['NPI'].replace(np.nan, df['NPI'].mean(), inplace = True)

####### Choosing Columns #######

removals = ['t.rfs', 'e.rfs', 't.os', 'e.os', 't.dmfs','e.dmfs', 't.tdm', 'e.tdm', "samplename", "id","filename","hospital", 'Surgery_type', 'Histtype', 'Angioinv', 'Lymp_infil', 'node', 'grade', 'er', 'risksg', 'risknpi', 'risk_AOL', 'veridex_risk']
rest = [i for i in df.columns if i not in removals]
X = df[rest]
y = df["t.tdm"]

####### Defining Indicator Functions #######

steps = 100
indicators = np.arange(0,10000,steps)

for indicator in indicators:
    df['indi_'+str(indicator)]= y < indicator
   
####### Train Test Split #######

X_train, X_test, y_train, y_test = train_test_split(X, df[['indi_'+str(indicator) for indicator in indicators]], test_size=0.33, random_state=42)
columns = y_train.columns

####### Logistic Regression #######
accs = []
for i,column in enumerate(columns):
    if( y_train[column].nunique()==1):
        y_test["prob_" + column] = 0
        y_test["survival_simple_" + column] = 1
        y_test["survival_log_" + column] = 0
        y_test["hazard_" + column] = 0
        continue
        
    clf = LogisticRegression(random_state=0,solver='liblinear')
    clf.fit(X_train, y_train[column])
    y_test["prob_" + column] = clf.predict_proba(X_test)[:,1]
    accs.append(clf.score(X_test, y_test[column]))
    if(i>1):
        y_test["prob_" + column] =  np.maximum(y_test["prob_" + columns[i-1]],y_test["prob_" + column])
        y_test["survival_simple_" + column] = 1 - y_test["prob_" + column]
        
        y_test["survival_log_" + column] = np.log(y_test["survival_simple_" + column])
        y_test["hazard_" + column] = ( - y_test["survival_log_" + columns[i]] + y_test["survival_log_" + columns[i-1]])/steps
print("Mean of the accuracies with Logistic Regression" , np.mean(accs))

####### Cobra Based Estimator #######
accs = []
for i,column in enumerate(columns):
    if( y_train[column].nunique()==1):
        y_test["prob_" + column] = 0
        y_test["survival_simple_" + column] = 1
        y_test["survival_log_" + column] = 0
        y_test["hazard_" + column] = 0
        continue
    cobra = Cobra(eps = 0.1)
    cobra = cobra.fit(X_train, y_train[column])
    preds = cobra.predict(X_test)
    preds[preds > 0.5] = 1
    preds[preds <= 0.5] = 0
    accs.append(accuracy_score(preds, y_test[column]))
#     y_test["prob_" + column] = preds
    if(i>1):
        y_test["prob_" + column] =  np.maximum(y_test["prob_" + columns[i-1]],y_test["prob_" + column])
        y_test["survival_simple_" + column] = 1 - y_test["prob_" + column]
        
        y_test["survival_log_" + column] = np.log(y_test["survival_simple_" + column])
        y_test["hazard_" + column] = ( - y_test["survival_log_" + columns[i]] + y_test["survival_log_" + columns[i-1]])/steps
print("Mean of the accuracies with Cobra Estimator" ,np.mean(accs)) 