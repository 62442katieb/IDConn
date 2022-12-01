import pandas as pd
from os.path import join, exists

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def impute(data, max_iter=10000):
    '''
    Fill in missing data with an iterative imputation algorithm from scikit learn.
    NOTE: Will not imput connectivity data.
    '''
    
    non_numeric = data.select_dtypes(exclude=['number']).columns
    dumb = pd.get_dummies(data[non_numeric], prefix='dummy')
    df = pd.concat([data.drop(non_numeric, axis=1), dumb])
    impute_pls = IterativeImputer(
        max_iter=max_iter, skip_complete=True, verbose=1, tol=5e-3, n_nearest_features=1000
    )
    imputed = impute_pls.fit_transform(df)
    imp_df = pd.DataFrame(imputed,columns=data.drop(non_numeric, axis=1).columns, index=data.index,
    )
    return imp_df

    
