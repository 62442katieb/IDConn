from idconn import nbs, io
import pandas as pd
import numpy as np
import bids
from os.path import join
from datetime import datetime
from time import strftime

today = datetime.today()
today_str = strftime("%m_%d_%Y")

TRAIN_DSET = '/Users/katherine.b/Dropbox/Data/diva-dset'
TEST_DSET = '/Users/katherine.b/Dropbox/Data/diva-dset'
DERIV_NAME = 'IDConn'
OUTCOME = 'Mean E2 (pg/mL)'
atlas_fname = '/Users/katherine.b/Dropbox/HPC-Backup-083019/physics-retrieval/craddock2012_tcorr05_2level_270_2mm.nii.gz'

layout = bids.BIDSLayout(TRAIN_DSET, derivatives=True)

dat = io.read_corrmats(layout, task='rest', atlas='craddock2012', z_score=False)

keep = dat['adj'].dropna().index
dat = dat.loc[keep]
#print(dat['adj'].values.shape)
num_node = dat.iloc[0]['adj'].shape[0]

matrices = np.vstack(dat['adj'].values).reshape((len(keep), num_node, num_node))
upper_tri = np.triu_indices(num_node, k=1)

outcome = np.reshape(dat[OUTCOME].values, (len(dat[OUTCOME]),1))
confounds = dat[['bc', 'menst_cycle-day']]
alpha = 0.1
fig_dir = '/Users/katherine.b/Dropbox/Projects/IDConn'

cv_results = nbs.kfold_nbs(matrices, outcome, confounds, alpha, tail='both', groups=None, n_splits=4, n_iterations=2, k=1000, shuffle=False, fig_dir=fig_dir)

cv_results.to_csv(join(TRAIN_DSET, 'derivatives', DERIV_NAME, f'nbs-predict__outcome-{OUTCOME}_models-{today_str}.tsv'),sep='\t')
best = cv_results[cv_results['score'] == cv_results['score'].max()].index[0]
subnetwork = cv_results.loc[best]['component']
subnetwork_df = pd.DataFrame(subnetwork,
                             index=range(0,num_node), 
                             columns=range(0,num_node))

subnetwork_df.to_csv(join(TRAIN_DSET, 'derivatives', DERIV_NAME, f'nbs-predict_edge_parameters-{today_str}.tsv'),sep='\t')

nbs_vector = subnetwork[upper_tri]
mask = nbs_vector == 1
edges = np.vstack(dat['edge_vector'].values)
features = edges[:,mask]
#plot the parameters
param_mat = cv_results.loc[best]['coefficient_matrix']
odds = 10 ** param_mat 
prob = odds / (1 + odds)

# run the model on the whole 28andMe dataset to get params
model = cv_results.loc[best]['model']
model.fit(features, outcome)
fig,fig2 = io.plot_edges(param_mat, atlas_fname, title=None, strength=True, cmap='icefire', node_size='strength')
fig.savefig('/Users/katherine.b/Dropbox/Projects/IDConn/test1.png')
fig2.savefig('/Users/katherine.b/Dropbox/Projects/IDConn/test2.png')