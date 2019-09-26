# Complete the function below.
import pandas as pd
import numpy as np
#from sklearn.metrics import r2_score
#from sklearn.datasets import load_boston
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
#from sklearn import svm
#import random
#import decimal 

#from math import sqrt
# 2, 3, 5, 7, 11, 13, 17, 19, 23 and 29.

import numpy as np

from sklearn import svm, datasets
from hyperopt import hp, tpe,  STATUS_OK, space_eval
from hyperopt import fmin
from hyperopt import Trials
from sklearn.model_selection import train_test_split,KFold, cross_val_score, GridSearchCV 


iris = datasets.load_iris()

print (iris.data),
print ( iris.target )


parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

space4svc = {'kernel': hp.choice('penalty', ['linear', 'rbf']),
            'C': hp.uniform('C', 1, 10) }

svc = svm.SVC(gamma="scale")
clf = GridSearchCV(svc, parameters, cv=5)

clf.fit(iris.data, iris.target)
'''                             
GridSearchCV(cv=5, error_score=...,
       estimator=SVC(C=1.0, cache_size=..., class_weight=..., coef0=...,
                     decision_function_shape='ovr', degree=..., gamma=...,
                     kernel='rbf', max_iter=-1, probability=False,
                     random_state=None, shrinking=True, tol=...,
                     verbose=False),
       iid=..., n_jobs=None,
       param_grid=..., pre_dispatch=..., refit=..., return_train_score=...,
       scoring=..., verbose=...)
'''
#print (sorted(clf.cv_results_.keys()) )
#print (clf.get_params() )
########################################

space4log = {'penalty': hp.choice('penalty', ['l1', 'l2']),'max_iter': hp.uniform('max_iter', 0, 200) }

X_train, X_test, y_train, y_test = train_test_split(iris.data,iris.target , test_size=0.33, random_state=42)

def hyperopt_train_test(params):
    #clf = LogisticRegression(**params)
    clf = svm.SVC(**params)
    return cross_val_score(clf, X_test, y_test).mean()# 78% => 95%


def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}# -95<-78

trials = Trials()

best = fmin(f, space4svc, algo=tpe.suggest, max_evals=100, trials=trials)

print(space_eval(space4svc, best))


