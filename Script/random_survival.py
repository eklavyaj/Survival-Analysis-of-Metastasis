import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
import matplotlib.pyplot as plt

df = pd.read_csv("../Data/selected.csv", index_col = False)
df.drop(["Unnamed: 0"], inplace = True, axis = 1)
df = df.replace(' NA', np.nan)
df = df.replace('NA', np.nan)
df['NPI'] = df['NPI'].astype('float')
df['NPI'].replace(np.nan, df['NPI'].mean(), inplace = True)

removals = ['t.rfs', 'e.rfs', 't.os', 'e.os', 't.dmfs','e.dmfs', 't.tdm', 'e.tdm', "samplename", "id","filename","hospital", 'Surgery_type', 'Histtype', 'Angioinv', 'Lymp_infil', 'node', 'grade', 'er', 'risksg', 'risknpi', 'risk_AOL', 'veridex_risk']
rest = [i for i in df.columns if i not in removals]

X = df[rest]
y = pd.DataFrame(df[['e.tdm', 't.tdm']])
y.columns = ['event', 'time']
y['event'] = y['event'] == 1
y = [tuple(row) for row in y.values]
out = np.empty(len(y), dtype =[('fstat', '?'), ('lenfol', '<f8')])
out[:] = y

X_train, X_test, y_train, y_test = train_test_split(X, out, test_size=0.50, random_state=42)

rsf = RandomSurvivalForest(n_estimators = 1000, 
                          min_samples_split=10, 
                          min_samples_leaf=15,
                          max_features="sqrt", 
                          random_state=0)
rsf.fit(X_train, y_train)
rsf.score(X_test, y_test)
chf = rsf.predict_cumulative_hazard_function(X_test, return_array=True)

print("Risk scores")
risks = [rsf.predict(X_test.iloc[[i]])[0] for i in range(len(X_test[::15]))]
index = [X_test.iloc[i].name for i in range(len(X_test[::15]))]
print(pd.Series(data=risks, index=index))

