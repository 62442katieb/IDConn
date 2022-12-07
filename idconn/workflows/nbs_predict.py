from idconn import nbs, io
import pandas as pd
import numpy as np
import bids
from os.path import join
from datetime import datetime
from time import strftime

today = datetime.today()
today_str = strftime("%m_%d_%Y")

TRAIN_DSET = '/Users/katherine.b/Dropbox/Data/ds002674'
TEST_DSET = '/Users/katherine.b/Dropbox/Data/diva-dset'
DERIV_NAME = 'IDConn'
OUTCOME = 'estradiol'
CONFOUNDS = ['bc']
TASK = 'rest'
ATLAS = 'craddock2012'
atlas_fname = '/Users/katherine.b/Dropbox/HPC-Backup-083019/physics-retrieval/craddock2012_tcorr05_2level_270_2mm.nii.gz'

layout = bids.BIDSLayout(TRAIN_DSET, derivatives=True)

dat = io.read_corrmats(layout, task=TASK, atlas=ATLAS, z_score=False)

keep = dat['adj'].dropna().index
dat = dat.loc[keep]
#print(dat['adj'].values.shape)
num_node = dat.iloc[0]['adj'].shape[0]

matrices = np.vstack(dat['adj'].values).reshape((len(keep), num_node, num_node))
upper_tri = np.triu_indices(num_node, k=1)

outcome = np.reshape(dat[OUTCOME].values, (len(dat[OUTCOME]),1))
confounds = dat[CONFOUNDS]
alpha = 0.1
fig_dir = '/Users/katherine.b/Dropbox/Projects/IDConn'

cv_results = nbs.kfold_nbs(matrices, outcome, confounds, alpha, tail='both', groups=None, n_splits=10, n_iterations=1000)

cv_results.to_csv(join(TRAIN_DSET, 'derivatives', DERIV_NAME, f'nbs-predict_outcome-{OUTCOME}_models-{today_str}.tsv'),sep='\t')
best = cv_results[cv_results['score'] == cv_results['score'].max()].index[0]
subnetwork = cv_results.loc[best]['component']
subnetwork_df = pd.DataFrame(subnetwork,
                             index=range(0,num_node), 
                             columns=range(0,num_node))

subnetwork_df.to_csv(join(TRAIN_DSET, 'derivatives', DERIV_NAME, f'nbs-predict__outcome-{OUTCOME}_edge-parameters-{today_str}.tsv'),sep='\t')

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
fig.savefig(join(TEST_DSET, 'derivatives', DERIV_NAME, f'nbs-predict_outcome-{OUTCOME}_betas-{today_str}.png'), dpi=400)
fig2.savefig(join(TEST_DSET, 'derivatives', DERIV_NAME, f'nbs-predict_outcome-{OUTCOME}_betas-strength-{today_str}.png'), dpi=400)

layout = bids.BIDSLayout(TEST_DSET, derivatives=True)

test_df = io.read_corrmats(layout, task=TASK, atlas=ATLAS, z_score=False)

test_df.dropna(inplace=True)

outcome_test = test_df[OUTCOME].values
groups_test = outcome
matrices_test = np.vstack(test_df['adj'].dropna().values).reshape((len(test_df['adj'].dropna().index),num_node,num_node))
edges_test = np.vstack(test_df['edge_vector'].dropna().values)

test_features = edges_test.T[mask,:]
test_outcome = test_df[OUTCOME].values
accuracy = model.score(test_features.T, test_outcome)
print('Independent prediction accuracy:\t', accuracy)
np.savetxt(join(TEST_DSET, 'derivatives', DERIV_NAME, f'nbs-predict__outcome-{OUTCOME}_accuracy-{today_str}.txt'), [accuracy])