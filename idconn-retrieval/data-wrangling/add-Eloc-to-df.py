import numpy as np
import pandas as pd

from os import makedirs
from os.path import join, exists


sink_dir = '/Users/kbottenh/Dropbox/Projects/physics-retrieval/data/output'
fig_dir = '/Users/kbottenh/Dropbox/Projects/physics-retrieval/figures/'
data_dir = '/Users/kbottenh/Dropbox/Projects/physics-retrieval/data'
roi_dir = '/Users/kbottenh/Dropbox/Data/templates/shen2015/'

masks = ['shen2015', 'craddock2012']

tasks = ['fci', 'retr']
sessions = [0, 1]
sesh = ['post']
conditions = ['high-level']
iqs = ['VCI', 'WMI', 'PRI', 'PSI', 'FSIQ']

# # Data wrangling
# Nodal efficiency data is currently in an <i>incredbily</i> long, multi-indexed dataframe. 
# Here, we transform it into wide data (dataframe per condition per task per session) for ease of analysis later.


df = pd.read_csv(join(data_dir, 'physics-learning-tasks_graphtheory_shen+craddock_nodal.csv'), index_col=[0,1,2,3,4], header=0)
df.rename({'Unnamed: 1': 'session', 'Unnamed: 2': 'task', 'Unnamed: 3': 'condition'}, axis=1, inplace=True)
null_df = pd.read_csv(join(sink_dir, 'local_efficiency', 'task_eff_dist.csv'), 
                      index_col=[0,1,2,3], header=0)

for i in np.arange(0,268)[::-1] :
    df.rename({'lEff{0}'.format(i): 'lEff{0}'.format(i+1)}, axis=1, inplace=True)

big_df = pd.read_csv(join(data_dir, 'rescored', 'physics_learning-nonbrain_OLS+fd-BayesianImpute.csv'), 
                index_col=0, header=0)

wide_effs = df.unstack(level=1).unstack(level=1).unstack(level=1).unstack(level=1)

pre_effs = list(wide_effs.filter(regex='(\'*\', 0.*)').columns)
ctrl_effs = list(wide_effs.filter(regex='(.* \'lower-level\'.*)').columns)
reas_effs = list(wide_effs.filter(regex='(.* \'reas\'.*)').columns)

drop = pre_effs + ctrl_effs + reas_effs
wide_effs.drop(drop, axis=1, inplace=True)

big_df = pd.concat([big_df, wide_effs], axis=1)

big_df.to_csv(join(data_dir, 'rescored', 'physics_learning-nonbrain_OLS+fd+local_efficiency.csv'))