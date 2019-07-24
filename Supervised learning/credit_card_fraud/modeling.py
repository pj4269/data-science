# https://www.kaggle.com/artgor/how-to-not-overfit
# https://www.kaggle.com/currie32/predicting-fraud-with-tensorflow
# https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, StratifiedKFold, KFold, cross_val_score, GridSearchCV, RepeatedStratifiedKFold

from functions import train_model
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from imblearn.over_sampling import SMOTE


#from sklearn.metrics import roc_auc_score


df = pd.read_csv(filepath_or_buffer = "/home/micah/Desktop/data/creditcard.csv")

print df.head()
# Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
 # 1. Create an evenly balanced training and test data

Fraud = df[df.Class == 1]
Normal = df[df.Class == 0]

# Set X_train equal to 80% of the fraudulent transactions.
X_train = Fraud.sample(frac=0.8)

# Add 80% of the normal transactions to X_train.
X_train = pd.concat([X_train, Normal.sample(frac = 0.8)], axis = 0)

# X_test contains all the transaction not in X_train.
X_test = df.loc[~df.index.isin(X_train.index)]

#Shuffle the dataframes so that the training is done in a random order.
X_train = shuffle(X_train)
X_test = shuffle(X_test)

y_train = X_train['Class']
y_test = X_test['Class']

X_train = X_train.drop('Class', axis=1)
X_test = X_test.drop('Class', axis=1)


print X_train.shape
print X_test.shape
print X_train.isnull().sum()
'''
#scaler = StandardScaler()
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
'''
print 'hi'
print X_train[:1]
# Standardization: Amount column
'''
scaler = StandardScaler()
X_ = scaler.fit_transform(X_train['Amount'].values.reshape(-1, 1))
X_test_amount = scaler.fit_transform(X_test['Amount'].values.reshape(-1, 1))

#X_test = scaler.transform(X_test)

X_train['Amount'] = pd.DataFrame(X_.ravel())
X_test['Amount'] = pd.DataFrame(X_test_amount.ravel())
'''
# Model

model = LogisticRegression(class_weight='balanced', penalty='l1', C=47, max_iter = 8)# 0.9461
#model = LogisticRegression(class_weight='balanced', penalty='l1', C=0.35, max_iter = 13)
#model = LogisticRegression(class_weight='balanced', penalty='l1', C=0.19, max_iter = 162)#0.9440


# Balancing the data
# a. Oversampling SMOTE
'''
sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))

#model.fit(X_train_res, y_train_res.ravel())
'''
# Cross validation
#folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)# 0.9463
#repeated_folds = RepeatedStratifiedKFold(n_splits=n_fold, n_repeats=5, random_state=42)# 0.9426
n_fold = 5
folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42) # 0.9424 => 0.9443 => 0.9456


'''
oof_lr, prediction_lr, scores = train_model(X_train_res, X_test.values, y_train_res, params=None, model_type='sklearn', model=model, folds = folds, n_fold = n_fold)# 0.9760
'''

# Ensembling: 
#bagging = BaggingClassifier(base_estimator=model, n_estimators=10, max_samples=0.8, max_features=0.8)

#oof_lr, prediction_lr, scores = train_model(X_train.values, X_test.values, y_train.values, params=None, model_type='sklearn', model=model, folds = folds, n_fold = n_fold)


# Param optimization
#lr = LogisticRegression(solver='liblinear', max_iter=100)
'''
# a.
# gridsearch => unnecessarily exhaustive
parameter_grid = {'class_weight' : ['balanced', None],
                  'penalty' : ['l2'],
                  'C' : [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                  'solver': ['newton-cg', 'sag', 'lbfgs']
                 }

grid_search = GridSearchCV(lr, param_grid=parameter_grid, cv=folds, scoring='roc_auc')
grid_search.fit(X_train, y_train)
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))
'''
'''
# b. hyperopt
def hyperopt_train_test(params):
    clf = LogisticRegression(**params)
    return cross_val_score(clf, X_test, y_test, scoring='roc_auc').mean()

space4log = {'penalty': hp.choice('penalty', ['l1', 'l2']),
            'max_iter': hp.uniform('max_iter', 0, 200), 
            'C' : hp.uniform('C', 0, 1 )
             
              }


def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space4log, algo=tpe.suggest, max_evals=100, trials=trials)
print 'best:'
print best  
'''

# feature selection

# a) Mlextend SequentialFeatureSelector
'''
sfs1 = SFS(model, k_features=(10, 15), forward=True, floating=False, verbose=0, scoring='roc_auc', cv=folds, n_jobs=-1)

sfs1 = sfs1.fit(X_train, y_train)

top_features = list(sfs1.k_feature_names_)
X_train_ = X_train[top_features]
X_test_ = X_test[top_features]
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

oof_lr, prediction_lr, _ = train_model(X_train_.values, X_test_.values, y_train.values, params=None, model_type='sklearn', model=model, folds = folds, n_fold = n_fold)# 0.9444
'''
# b) sklearn feature selection

'''
selector = SelectKBest(f_classif, k=10)
X_trainK = selector.fit_transform(X_train.values, y_train.values)
X_testK = selector.transform(X_test.values)
oof_lr_1, prediction_lr_1, scores = train_model(X_trainK, X_testK, y_train.values, params=None, model_type='sklearn', model=model, folds = folds, n_fold = n_fold) # 0.9402
'''

# c) SHAP
'''
explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train)
'''
# d) ELI5
import eli5

oof_lr, prediction_lr, _ = train_model(X_train.values, X_test.values, y_train.values, params=None, model_type='sklearn', model=model, folds = folds, n_fold = n_fold)
print eli5.show_weights(model, top=10)

