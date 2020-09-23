import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
#####################################################################################################
# reference
X = np.array([[10, 20, 15], [30, 40, 25], [50, 60, 55], [70, 80, 75]])
y = np.array([100, 200, 300, 400])
kf = KFold(n_splits=2)
kf.get_n_splits(X)
print X, y

df = pd.DataFrame(data=X)
#print df

#print df.iloc[[1, 2]][0]# 1,2nd rows and 0 col

#KFold(n_splits=2, random_state=None, shuffle=False)
#for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    #X_train, X_test = X[train_index], X[test_index]
    #print train_index, test_index
    #y_train, y_test = y[train_index], y[test_index]

######################################################################################################


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# new
train_data =  train_data.head(n=4)
test_data =  test_data.head(n=4)# test is just a rows that are missing fron train
#print train_data
#print test_data

features = train_data.columns[2:]


numeric_features = []
categorical_features = []

for dtype, feature in zip(train_data.dtypes[2:], train_data.columns[2:]):
    if dtype == object:
        #print(column)
        #print(train_data[column].describe())
        categorical_features.append(feature)
    else:
        numeric_features.append(feature)
#print categorical_features


np.random.seed(13)

def impact_coding(data, feature, target='y'):
    #In this implementation we get the values and the dictionary as two different steps.
    #This is just because initially we were ignoring the dictionary as a result variable.
    #In this implementation the KFolds use shuffling. If you want reproducibility the cv 
    #could be moved to a parameter.
    n_folds = 2 
    n_inner_folds = 2
    impact_coded = pd.Series()
    
    oof_default_mean = data[target].mean() # Gobal mean to use by default (you could further tune this)
    
    kf = KFold(n_splits=n_folds, shuffle=True)
    oof_mean_cv = pd.DataFrame()
    split = 0
    
    for infold, oof in kf.split(data[feature]):# train on infold and test on oof # ref: https://machinelearningmastery.com/k-fold-cross-validation/
            impact_coded_cv = pd.Series()
            kf_inner = KFold(n_splits=n_inner_folds, shuffle=True)
            inner_split = 0
            inner_oof_mean_cv = pd.DataFrame()
            oof_default_inner_mean = data.iloc[infold][target].mean()# mean for the selected target within the split
 
            for infold_inner, oof_inner in kf_inner.split(data.iloc[infold]):# train on infold_inner and test on oof_inner
                # infold is further split into inner infold and oof
                print infold_inner, oof_inner 
                # target mean for each inner_folder for the training
                oof_mean = data.iloc[infold_inner].groupby(by=feature)[target].mean()
                print data.iloc[infold_inner]
                print 'oof_mean: ',oof_mean
                print 'oof_mean.index: ',oof_mean.index
                impact_coded_cv = impact_coded_cv.append(data.iloc[infold].apply(
                            lambda x: oof_mean[x[feature]]
                                      if x[feature] in oof_mean.index
                                      else oof_default_inner_mean
                            , axis=1))

                # Also populate mapping (this has all group -> mean for all inner CV folds) 
                # rsuffix: add inner_split at the end of the key joining from the right
                #  https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html
                inner_oof_mean_cv = inner_oof_mean_cv.join(pd.DataFrame(oof_mean), rsuffix=inner_split, how='outer')
   # print inner_oof_mean_cv 
    
                inner_oof_mean_cv.fillna(value=oof_default_inner_mean, inplace=True)
                inner_split += 1

            # Also populate mapping
            oof_mean_cv = oof_mean_cv.join(pd.DataFrame(inner_oof_mean_cv), rsuffix=split, how='outer')
            oof_mean_cv.fillna(value=oof_default_mean, inplace=True)
            split += 1
            
            impact_coded = impact_coded.append(data.iloc[oof].apply(
                            lambda x: inner_oof_mean_cv.loc[x[feature]].mean()
                                      if x[feature] in inner_oof_mean_cv.index
                                      else oof_default_mean
                            , axis=1))

    return impact_coded, oof_mean_cv.mean(axis=1), oof_default_mean



def impact_coding(data, feature, target='y'):# y is the target col name
   
    #In this implementation we get the values and the dictionary as two different steps.
    #This is just because initially we were ignoring the dictionary as a result variable.
    
    #In this implementation the KFolds use shuffling. If you want reproducibility the cv 
    #could be moved to a parameter.
    
    n_folds = 2
    n_inner_folds = 2
    impact_coded = pd.Series()
    
    oof_default_mean = data[target].mean() # Target col mean
    kf = KFold(n_splits=n_folds, shuffle=True)
    oof_mean_cv = pd.DataFrame()
    split = 0
    #print target
    #print data[target]
    #print oof_default_mean
    for infold, oof in kf.split(data[feature]):
        impact_coded_cv = pd.Series()
        kf_inner = KFold(n_splits=n_inner_folds, shuffle=True)
        inner_split = 0
        inner_oof_mean_cv = pd.DataFrame()
        oof_default_inner_mean = data.iloc[infold][target].mean()# mean of the target values for the training set
                                                                 # infold= row numbers, target=col number=> gives you specific target values with certain row numbers(infold). and find the mean of that! Look above for reference!
                                                                 # here its a training target mean!
        #print 'data[feature]: ',data[feature]
        #print 'infold= rows for training: ', infold
        #print 'oof = rows for testing', oof        
        #print 'target: ', target     
        #print 'data.iloc[infold][target]', data.iloc[infold][target]

        # training set it being split again into inner training and inner test sets!
        for infold_inner, oof_inner in kf_inner.split(data.iloc[infold]):
            # The mean to apply to the inner oof split (a 1/n_folds % based on the rest)
            #print 'infold_inner ',infold_inner
            #print 'feature ', feature
            oof_mean = data.iloc[infold_inner].groupby(by=feature)[target].mean()# target mean for row numbers: infold_inner(e.g. [1, 2,5])
            #print 'data.iloc[infold_inner]',data.iloc[infold_inner]
            #print 'data.iloc[infold_inner].groupby(by=feature)[target] ',data.iloc[infold_inner].groupby(by=feature)[target]
            #print 'oof_mean ',oof_mean
            #print 'oof_mean.index', oof_mean.index

            impact_coded_cv = impact_coded_cv.append(data.iloc[infold].apply(lambda x: oof_mean[x[feature]] if x[feature] in oof_mean.index   
                                    else oof_default_inner_mean, axis=1))
            #print 'impact_coded_cv: ',impact_coded_cv
            inner_oof_mean_cv = inner_oof_mean_cv.join(pd.DataFrame(oof_mean), rsuffix=inner_split, how='outer')
            inner_oof_mean_cv.fillna(value=oof_default_inner_mean, inplace=True)
            inner_split += 1

            # Also populate mapping
            oof_mean_cv = oof_mean_cv.join(pd.DataFrame(inner_oof_mean_cv), rsuffix=split, how='outer')
            oof_mean_cv.fillna(value=oof_default_mean, inplace=True)
            split += 1
            
            impact_coded = impact_coded.append(data.iloc[oof].apply(
                            lambda x: inner_oof_mean_cv.loc[x[feature]].mean()
                                      if x[feature] in inner_oof_mean_cv.index
                                      else oof_default_mean
                            , axis=1))

    return impact_coded, oof_mean_cv.mean(axis=1), oof_default_mean



for f in categorical_features:
    print("Impact coding for {}".format(f))
    print impact_coding(train_data, f)
    #print train_data.head()
print train_data[categorical_features]

impact_coding_map = {}
#train_data.iloc[:,:10]
for f in categorical_features:
    print("Impact coding for {}".format(f))
    train_data["impact_encoded_{}".format(f)], mapping, default_mean = impact_coding(train_data, f)
    #train_data["impact_encoded_{}".format(f)], b,c = impact_coding(train_data, f)
    #impact_coding_map[f] = (impact_coding_mapping, default_coding)
    #mapping, default_mean = impact_coding_map[f]
    print mapping
    test_data["impact_encoded_{}".format(f)] = test_data.apply(lambda x: mapping[x[f]]
                                                                         if x[f] in mapping
                                                                         else default_mean
                                                               , axis=1)



'''
print default_mean
print mapping

print train_data.iloc[:,:10]
print train_data.iloc[:,-10:]
print test_data.iloc[:,:10]
print test_data.iloc[:,-10:]
'''
