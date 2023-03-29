#!/usr/bin/env python3
import pandas as pd
import numpy as np
import pingouin as pg
import nibabel as nib
import bids
from os.path import join
from datetime import datetime
from time import strftime
from scipy.stats import spearmanr
from idconn import nbs, io


from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import warnings
import json

warnings.simplefilter("ignore")

today = datetime.today()
today_str = strftime("%m_%d_%Y")

TRAIN_DSET = '/Users/katherine.b/Dropbox/Data/ds002674'
TEST_DSET = '/Users/katherine.b/Dropbox/Data/diva-dset'
DERIV_NAME = 'IDConn'
OUTCOME = 'bc'
CONFOUNDS = 'fd'
TASK = 'rest'
ATLAS = 'craddock2012'
alpha = 0.05
atlas_fname = '/Users/katherine.b/Dropbox/HPC-Backup-083019/physics-retrieval/craddock2012_tcorr05_2level_270_2mm.nii.gz'


layout = bids.BIDSLayout(TRAIN_DSET, derivatives=True)

dat = io.read_corrmats(layout, task=TASK, deriv_name='IDConn', atlas=ATLAS, z_score=True)

keep = dat['adj'].dropna().index
dat = dat.loc[keep]
#print(dat['adj'].values.shape)
num_node = dat.iloc[0]['adj'].shape[0]

matrices = np.vstack(dat['adj'].values).reshape((len(keep), num_node, num_node))
upper_tri = np.triu_indices(num_node, k=1)

outcome = np.reshape(dat[OUTCOME].values, (len(dat[OUTCOME]),1))

if CONFOUNDS is not None:
    confounds = dat[CONFOUNDS]
else:
    confounds = None
#print(dat['bc'])

weighted_average, cv_results = nbs.kfold_nbs(matrices, outcome, confounds, alpha, groups=dat['bc'], n_splits=10, n_iterations=100)

fig,fig2, nimg = io.plot_edges(weighted_average, 
                         atlas_fname, 
                         threshold='computed', 
                         title=f'{OUTCOME} Precition-Weighted Average', 
                         strength=True, 
                         cmap='seismic', 
                         node_size='strength')

if CONFOUNDS is not None:
    base_name = f'nbs-predict_outcome-{OUTCOME}_confounds-{CONFOUNDS}'
else:
    base_name = f'nbs-predict_outcome-{OUTCOME}'

fig.savefig(join(TRAIN_DSET, 'derivatives', DERIV_NAME, f'{base_name}_weighted-{today_str}.png'), dpi=400)
fig2.savefig(join(TRAIN_DSET, 'derivatives', DERIV_NAME, f'{base_name}_weighted-strength-{today_str}.png'), dpi=400)
nib.save(nimg, join(TRAIN_DSET, 'derivatives', DERIV_NAME, f'{base_name}_weighted-strength-{today_str}'))


avg_df = pd.DataFrame(weighted_average, 
                      index=range(0,weighted_average.shape[0]),
                      columns=range(0,weighted_average.shape[1]))

cv_results.to_csv(join(TRAIN_DSET, 'derivatives', DERIV_NAME, f'{base_name}_models-{today_str}.tsv'),sep='\t')
avg_df.to_csv(join(TRAIN_DSET, 'derivatives', DERIV_NAME, f'{base_name}_weighted-{today_str}.tsv'),sep='\t')


# this uses the most predictive subnetwork as features in the model
# might replace with thresholded weighted_average
# or use _all_ the edges in weighted_average with KRR or ElasticNet...
# ORRR use thresholded weighted average edges with ElasticNet...
# - stays true to NBS-Predict
# - increases parsimony while handling multicollinearity...
# either way, I don't think cv_results is necessary

#best = cv_results[cv_results['score'] == cv_results['score'].max()].index[0]
#subnetwork = cv_results.loc[best]['component']
#subnetwork_df = pd.DataFrame(subnetwork,
#                             index=range(0,num_node), 
#                             columns=range(0,num_node))

#if CONFOUNDS is not None:
#    subnetwork_df.to_csv(join(TRAIN_DSET, 'derivatives', DERIV_NAME, f'nbs-predict_outcome-{OUTCOME}_confounds-{CONFOUNDS}_edge-parameters-{today_str}.tsv'),sep='\t')
#else:
#    subnetwork_df.to_csv(join(TRAIN_DSET, 'derivatives', DERIV_NAME, f'nbs-predict_outcome-{OUTCOME}_edge-parameters-{today_str}.tsv'),sep='\t')

# here is where we'd threshold the weighted average to use for elastic-net

nbs_vector = weighted_average[upper_tri]
p50 = np.percentile(nbs_vector, 50)
filter = np.where(nbs_vector >= p50, True, False)
#print(nbs_vector.shape, filter.shape)

#mask = io.vectorize_corrmats(filter)
edges_train = np.vstack(dat['edge_vector'].dropna().values)

#print(features.shape)

scaler = StandardScaler()
edges_train = scaler.fit_transform(edges_train)
if len(np.unique(outcome)) <= 2:
    pass
else:
    outcome = scaler.fit_transform(outcome)


#edges = np.vstack(dat['edge_vector'].values)
#features = edges[:,mask]

# NEED TO RESIDUALIZE IF CONFOUNDS IS NOT NONE
if CONFOUNDS is not None:
    #regress out the confounds from each edge and the outcome variable, 
    # use the residuals for the rest of the algorithm
    #print(confounds.shape, outcome.shape)
    outcome_train = np.reshape(outcome, (outcome.shape[0],))
    y = pg.linear_regression(confounds, outcome_train)
    train_outcome = y.residuals_

    resid_edges = np.zeros_like(edges_train)
    for i in range(0, edges_train.shape[1]):
        x = pg.linear_regression(confounds, edges_train[:,i])
        resid_edges[:,i] = x.residuals_
    train_features = resid_edges[:,filter]
else:
    train_features = edges_train[:,filter]
    train_outcome = outcome

# run the model on the whole test dataset to get params

# classification if the outcome is binary (for now)
# could be extended to the multiclass case?

if len(np.unique(outcome)) == 2:
    model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.25, warm_start=True)
else:
    model = ElasticNet(l1_ratio=0.25, warm_start=True)

# train ElasticNet on full train dataset, using feature extraction from NBS-Predict
train_metrics = {}
fitted = model.fit(X=train_features, y=np.ravel(train_outcome))
in_sample_score = fitted.score(X=train_features, y=np.ravel(train_outcome))
if len(np.unique(outcome)) == 2:
    train_metrics['accuracy'] = in_sample_score
else:
    train_metrics['coefficient of determination'] = in_sample_score
y_pred = fitted.predict(X=train_features)
mse = mean_squared_error(train_outcome, y_pred)
train_metrics['mean squared error'] = mse
print('In-sample prediction score: ', in_sample_score)
print('In-sample mean squared error: ', mse)
#print(np.mean(train_features))
with open(join(TRAIN_DSET, 'derivatives', DERIV_NAME, f'{base_name}_fit-{today_str}.json'), 'w') as fp:
    json.dump(train_metrics, fp)



# yoink the coefficients? for a more parsimonious figure?
coeff_vec = np.zeros_like(filter)
j = 0
for i in range(0, filter.shape[0]):
    if filter[i] == True:
        if len(np.unique(outcome)) == 2:
            coeff_vec[i] = fitted.coef_[0,j]
        else:
            coeff_vec[i] = fitted.coef_[j]
        j += 1
    else:
        pass

#print(coeff_vec)

coef_mat = io.undo_vectorize(coeff_vec, num_node=num_node)
#print(coef_mat == coef_mat.T)

fig,fig2, nimg = io.plot_edges(coef_mat, 
                         atlas_fname, 
                         threshold='computed',
                         title=f'{OUTCOME} Coefficients', 
                         strength=True, 
                         cmap='seismic', 
                         node_size='strength')

fig.savefig(join(TRAIN_DSET, 'derivatives', DERIV_NAME, f'{base_name}_betas-{today_str}.png'), dpi=400)
fig2.savefig(join(TRAIN_DSET, 'derivatives', DERIV_NAME, f'{base_name}_betas-strength-{today_str}.png'), dpi=400)
nib.save(nimg, join(TRAIN_DSET, 'derivatives', DERIV_NAME, f'{base_name}_betas-strength-{today_str}'))


layout = bids.BIDSLayout(TEST_DSET, derivatives=True)

test_df = io.read_corrmats(layout, task=TASK, deriv_name='IDConn', atlas=ATLAS, z_score=True)

keep = test_df[[OUTCOME, 'adj']].dropna().index
#print(keep)

test_df = test_df.loc[keep]

outcome_test = test_df[OUTCOME].values

if len(np.unique(outcome_test)) <= 2:
    pass
else:
    outcome_test = scaler.fit_transform(outcome_test.reshape(-1, 1))

#print(outcome_test)
matrices_test = np.vstack(test_df['adj'].dropna().values).reshape((len(test_df['adj'].dropna().index),num_node,num_node))
edges_test = np.vstack(test_df['edge_vector'].dropna().values)
edges_test = scaler.fit_transform(edges_test)

# NEED TO RESIDUALIZE IF CONFOUNDS IS NOT NONE
if confounds is not None:
    confounds_test = test_df[CONFOUNDS].values
    #regress out the confounds from each edge and the outcome variable, 
    # use the residuals for the rest of the algorithm
    #print(confounds.shape, outcome.shape)
    outcome_test = np.reshape(outcome_test, (outcome_test.shape[0],))
    y = pg.linear_regression(confounds_test, outcome_test)
    test_outcome = y.residuals_

    resid_edges = np.zeros_like(edges_test)
    for i in range(0, edges_test.shape[1]):
        x = pg.linear_regression(confounds_test, edges_test[:,i])
        resid_edges[:,i] = x.residuals_
    test_features = resid_edges[:,filter]
else:
    test_features = edges_test[:,filter]
    test_outcome = outcome_test

#print(test_features.shape)
# if the model is a logistic regression, i.e. with a binary outcome
# then score is prediction accuracy
# if the model is a linear regression, i.e., with a continuous outcome
# then the score is R^2 (coefficient of determination)

# fit trained ElasticNet, initialized via warm_start
# prob in CV?
#fitted_test = fitted.fit(X=test_features, y=np.ravel(test_outcome))
#score = fitted_test.score(X=test_features, y=np.ravel(test_outcome))
test_metrics = {}
y_pred = fitted.predict(X=test_features)
score = fitted.score(X=test_features, y=np.ravel(test_outcome))
if len(np.unique(test_outcome)) == 2:
    test_metrics['accuracy'] = score
else:
    test_metrics['coefficient of determination'] = score
mse = mean_squared_error(test_outcome, y_pred)
test_metrics['mean squared error'] = mse
print('Out-of-sample prediction score:\t', score)
print('Out-of-sample mean squared error:\t', mse)
#print(np.mean(test_features))
pred_outcome = fitted.predict(test_features)

#print(test_outcome, '\n',pred_outcome)
#print(pred_outcome)
if len(np.unique(test_outcome)) > 2:
    corr = spearmanr(test_outcome, pred_outcome)
    print(f'\nSpearman correlation between predicted and actual {OUTCOME}:\t', corr)
    test_metrics['spearman correlation'] = corr
with open(join(TEST_DSET, 'derivatives', DERIV_NAME, f'{base_name}_fit-{today_str}.json'), 'w') as fp:
    json.dump(test_metrics, fp)
