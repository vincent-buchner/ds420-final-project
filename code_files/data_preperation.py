from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
import numpy as np 

def prepare_for_train(dftrain, dftest):
   
    num_pipeline = make_pipeline(SimpleImputer(strategy = 'median'), 
                                 StandardScaler())
    
    #cat_pipeline = make_pipeline(SimpleImputer(strategy = 'most_frequent'))
    
    full_pipeline = make_pipeline(ColumnTransformer([
        ("num", num_pipeline, make_column_selector(dtype_include=np.number)),
        #("cat", cat_pipeline, make_column_selector(dtype_include=object))
        ]))
    
    Xtrain, ytrain = dftrain.drop(columns = ['Amount']), dftrain['Amount']
    Xtest, ytest = dftest.drop(columns = ['Amount']), dftest['Amount']

    Xtrain_prepared = full_pipeline.fit_transform(Xtrain)
    Xtest_prepared = full_pipeline.fit_transform(Xtest)
    
    Xtrain_prepared, ytrain_prepared = handle_outlier(Xtrain_prepared, ytrain)

    return Xtrain_prepared, ytrain_prepared, Xtest_prepared, ytest


def handle_outlier(X, y):
    outlier_ind = get_outlier_indices(X)
    return X[outlier_ind == 1], y[outlier_ind == 1]

def get_outlier_indices(X):  
    model = LocalOutlierFactor()
    return model.fit_predict(X)