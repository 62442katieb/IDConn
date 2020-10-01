import numpy as np
import pandas as pd
from os import makedirs
from os.path import join, exists


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import datetime

sink_dir = '/Users/kbottenh/Dropbox/Projects/physics-retrieval/data/output'
fig_dir = '/Users/kbottenh/Dropbox/Projects/physics-retrieval/figures/'
data_dir = '/Users/kbottenh/Dropbox/Projects/physics-retrieval/data'

b_df = pd.read_csv(
    join(data_dir, 'rescored', 'physics_learning-nonbrain_OLS+fd-BayesianImpute.csv'), index_col=0, header=0)

gt_df = pd.read_csv(join(sink_dir, 'physics-learning-tasks_graphtheory_shen+craddock.csv'), 
                    index_col=[0,1,2,3,4], header=0)

gt_df.drop(['assortativity', 'transitivity', 'charpath', 'modularity'], axis=1, inplace=True)


wide_eff = gt_df.unstack(level=1).unstack(level=1).unstack(level=1).unstack(level=1)

big_df = pd.concat([b_df, wide_eff], axis=1)

big_df.to_csv(join(data_dir, 'rescored', 'physics_learning-nonbrain_OLS+fd+eff-missing.csv'))

impute_pls = IterativeImputer(max_iter=10000, skip_complete=True, verbose=1, tol=1e-3, n_nearest_features=100)
imputed = impute_pls.fit_transform(big_df.drop(['Sex', 'Class.Type'], axis=1))
imp_df = pd.DataFrame(imputed, 
                      columns=big_df.drop(['Sex', 'Class.Type'], axis=1).columns, 
                      index=big_df.index)
imp_df['Sex'] = big_df['Sex']
imp_df['Class.Type'] = big_df['Class.Type']


null_df = pd.read_csv(join(sink_dir, 'local_efficiency', 'task_eff_dist.csv'), 
                      index_col=[0,1,2,3], header=0)

masks = ['shen2015', 'craddock2012']
tasks = ['fci', 'retr']
sessions = [0, 1]
sesh = ['pre', 'post']
conditions = ['high-level', 'lower-level']
iqs = ['VCI', 'WMI', 'PRI', 'PSI', 'FSIQ']

for mask in masks:
    for session in sessions:
        for task in tasks:
            for condition in conditions:
                if condition == 'high-level':
                    cond = 'physics'
                elif condition == 'lower-level':
                    cond = 'control'
                conns = big_df.filter(regex='(\'*\', {0}, \'{1}\', \'{2}\', \'{3}\')'.format(session, 
                                                                                             task, 
                                                                                             condition, 
                                                                                             mask)).columns
                new_conns = []
                for conn in conns:
                    new_conn = '_'.join(str(s) for s in conn)
                    new_conns.append(new_conn)
                    big_df.rename({conn: new_conn}, axis=1, inplace=True)

                big_df[new_conns] = big_df[new_conns] / null_df.loc[sesh[session], 
                                                            task, 
                                                            cond, 
                                                            mask]['mean']

drop = big_df.filter(regex='.*reas.*').columns
big_df.drop(drop, axis=1, inplace=True)

big_df.to_csv(join(data_dir, 'rescored', 'physics_learning-nonbrain_OLS+fd+eff-BayesianImpute.csv'))
