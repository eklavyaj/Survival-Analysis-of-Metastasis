from sklearn.base import BaseEstimator
from sklearn import neighbors, tree, svm
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, SGDClassifier
from math import fabs
import numpy as np


models = {"svc": svm.SVC, "knn": neighbors.KNeighborsClassifier, "dt": tree.DecisionTreeClassifier, "logreg": LogisticRegression, "gaussnb": GaussianNB, "lda": LinearDiscriminantAnalysis}


class Cobra(BaseEstimator):
    def __init__(self, eps=None):
        self.eps = eps
        self.machines = {}
        self.models = {'svc': svm.SVC,
                     'knn': neighbors.KNeighborsClassifier,
                     'dt': tree.DecisionTreeClassifier,
                     'logreg': LogisticRegression,
                     'gaussnb': GaussianNB,
                     'lda': LinearDiscriminantAnalysis}
        self.alpha = len(self.models)
    
    def fit(self, X_train, y_train):
        # Fitting each of the models
        for model_name in self.models:
            if model_name == "logreg":
                self.machines[model_name] = models[model_name](solver = 'liblinear').fit(X_train, y_train)
            elif model_name =='svc':
                self.machines[model_name] = models[model_name](probability = True).fit(X_train, y_train)
            else:
                self.machines[model_name] = models[model_name]().fit(X_train, y_train)
        
        return self
    def predict(self, X_test):
        # Genarating Predictions from trained regressors.
        self.machine_predictions_ = {}
        self.all_predictions_ = []
        for machine_name in self.machines:
            self.machine_predictions_[machine_name] = self.machines[machine_name].predict_proba(X_test)[:,1]
            self.all_predictions_.append(self.machine_predictions_[machine_name])
        self.all_predictions_= np.array(self.all_predictions_)
        selection_table = np.zeros((len(X_test), self.alpha))
        
        for machine_index, machine_name in enumerate(self.models.keys()):
            predicted_val = self.machines[machine_name].predict_proba(X_test)[:,1]

            for ind in range(0, len(X_test)):
                if fabs(self.machine_predictions_[machine_name][ind] - predicted_val[ind]) <= self.eps:
                    selection_table[ind][machine_index] = 1


        
        predicted_vals = []
        for ind in range(len(X_test)):
            if(np.sum(selection_table[ind]) == self.alpha):
                predicted_vals.append(np.mean(self.all_predictions_[:, ind]))
                
            else:
                predicted_vals.append(0)
        return np.array(predicted_vals)