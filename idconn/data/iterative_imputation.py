import pandas as pd
from os.path import join, exists

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


#sink_dir = "/Users/kbottenh/Dropbox/Projects/physics-retrieval/data/rescored"
# sink_dir = '/home/kbott006/physics-retrieval'
# fig_dir = '/Users/kbottenh/Dropbox/Projects/physics-retrieval/figures/'
#data_dir = "/Users/kbottenh/Dropbox/Projects/physics-retrieval/data/rescored"
# roi_dir = '/Users/kbottenh/Dropbox/Data/templates/shen2015/'
# data_dir = '/home/kbott006/physics-retrieval'

# big_df = pd.read_csv(join(data_dir, 'physics_learning-nonbrain_OLS-missing+fd+local_efficiency.csv'),
#                index_col=0, header=0)

# impute first?
def impute(data, max_iter):
    non_numeric = data.select_dtypes(exclude=['number']).columns
    dumb = pd.get_dummies(data[non_numeric], prefix='dummy')
    df = pd.concat([data.drop(non_numeric, axis=1), dumb])
    impute_pls = IterativeImputer(
        max_iter=10000, skip_complete=True, verbose=1, tol=5e-3, n_nearest_features=1000
    )
    imputed = impute_pls.fit_transform(df)
    imp_df = pd.DataFrame(
        imputed,
        columns=data.drop(non_numeric, axis=1), axis=1).columns,
        index=data.index,
    )
    return imp_df

    
