#!/usr/bin/env python3
import pandas as pd
import numpy as np
import nibabel as nib
import seaborn as sns
import bids
import matplotlib.pyplot as plt
from os.path import join
from datetime import datetime
from time import strftime
from scipy.stats import spearmanr
from idconn import nbs, io

from bct import threshold_proportional


from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold, cross_validate
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.metrics import mean_squared_error
from matplotlib.colors import ListedColormap
import matplotlib as mpl


import warnings
import json

warnings.simplefilter("ignore")

today = datetime.today()
today_str = strftime("%m_%d_%Y")

TRAIN_DSET = "/Users/katherine.b/Dropbox/Data/ds002674"
TEST_DSET = "/Users/katherine.b/Dropbox/Data/diva-dset"
DERIV_NAME = "IDConn"
OUTCOME = "estradiol"
CONFOUNDS = ["framewise_displacement"]
TASK = "rest"
ATLAS = "craddock2012"
THRESH = 0.5
alpha = 0.01
atlas_fname = "/Users/katherine.b/Dropbox/HPC-Backup-083019/physics-retrieval/craddock2012_tcorr05_2level_270_2mm.nii.gz"


layout = bids.BIDSLayout(TRAIN_DSET, derivatives=True)

dat = io.read_corrmats(layout, task=TASK, deriv_name="IDConn", atlas=ATLAS, z_score=False)

drop = dat[dat["cycle_day"].between(11, 17, inclusive="neither")].index
keep = dat["adj"].dropna().index
dat = dat.loc[keep]

groups = dat["bc"]
# print(dat['adj'].values.shape)
num_node = dat.iloc[0]["adj"].shape[0]

matrices = np.vstack(dat["adj"].values).reshape((len(keep), num_node, num_node))
upper_tri = np.triu_indices(num_node, k=1)

outcome = np.reshape(dat[OUTCOME].values, (len(dat[OUTCOME]), 1))

# print(len(np.unique(outcome)))

if CONFOUNDS is not None:
    confounds = dat[CONFOUNDS]
    base_name = f"nbs-predict_outcome-{OUTCOME}_confounds-{CONFOUNDS}"
else:
    confounds = None
    base_name = f"nbs-predict_outcome-{OUTCOME}"
# print(dat['bc'])

weighted_average, cv_results = nbs.kfold_nbs(
    matrices, outcome, confounds, alpha, groups=groups, n_splits=5, n_iterations=500
)

fig, fig2, nimg = io.plot_edges(
    weighted_average,
    atlas_fname,
    threshold="computed",
    title=f"{OUTCOME} Precision-Weighted Average",
    strength=True,
    cmap="seismic",
    node_size="strength",
)

fig.savefig(
    join(TRAIN_DSET, "derivatives", DERIV_NAME, f"{base_name}_weighted-{today_str}.png"), dpi=400
)
fig2.savefig(
    join(TRAIN_DSET, "derivatives", DERIV_NAME, f"{base_name}_weighted-strength-{today_str}.png"),
    dpi=400,
)
nib.save(
    nimg, join(TRAIN_DSET, "derivatives", DERIV_NAME, f"{base_name}_weighted-strength-{today_str}")
)


avg_df = pd.DataFrame(
    weighted_average,
    index=range(0, weighted_average.shape[0]),
    columns=range(0, weighted_average.shape[1]),
)

cv_results.to_csv(
    join(TRAIN_DSET, "derivatives", DERIV_NAME, f"{base_name}_models-{today_str}.tsv"), sep="\t"
)
avg_df.to_csv(
    join(TRAIN_DSET, "derivatives", DERIV_NAME, f"{base_name}_weighted-{today_str}.tsv"), sep="\t"
)

best = cv_results.sort_values(by="score", ascending=False).iloc[0]["model"]

# this uses the most predictive subnetwork as features in the model
# might replace with thresholded weighted_average
# or use _all_ the edges in weighted_average with KRR or ElasticNet...
# ORRR use thresholded weighted average edges with ElasticNet...
# - stays true to NBS-Predict
# - increases parsimony while handling multicollinearity...
# either way, I don't think cv_results is necessary

# here is where we'd threshold the weighted average to use for elastic-net
weighted_average = np.where(weighted_average > 0, weighted_average, 0)
# print(np.sum(weighted_average))
# nbs_vector = weighted_average[upper_tri]
# p75 = np.percentile(nbs_vector, 75)
# filter = np.where(nbs_vector >= p75, True, False)
# print(np.sum(filter))
# print(nbs_vector.shape, filter.shape)

thresh_average = threshold_proportional(weighted_average, THRESH)
nbs_vector2 = thresh_average[upper_tri]
# p75 = np.percentile(nbs_vector, 75)
filter = np.where(nbs_vector2 > 0, True, False)

# mask = io.vectorize_corrmats(filter)
edges_train = np.vstack(dat["edge_vector"].dropna().values)[:, filter]

# NEED TO RESIDUALIZE IF CONFOUNDS IS NOT NONE
if CONFOUNDS is not None:
    confounds_train = dat[CONFOUNDS].values
    outcome_train = np.reshape(outcome, (outcome.shape[0],))
    # regress out the confounds from each edge and the outcome variable,
    # use the residuals for the rest of the algorithm
    # print(confounds.shape, outcome.shape)
    if len(np.unique(outcome_train)) <= 2:
        resid_edges = nbs.residualize(X=edges_train, confounds=confounds_train)
        train_outcome = outcome
    elif len(np.unique(outcome_train)) > 3:
        train_outcome, resid_edges = nbs.residualize(
            X=edges_train, y=outcome_train, confounds=confounds_train
        )
    train_features = resid_edges
else:
    train_features = edges_train
    train_outcome = outcome

x_scaler = StandardScaler()
y_scaler = StandardScaler()
train_features = x_scaler.fit_transform(train_features)
if len(np.unique(train_outcome)) <= 2:
    pass
else:
    train_outcome = y_scaler.fit_transform(train_outcome.reshape(-1, 1))


# run the model on the whole test dataset to get params

# classification if the outcome is binary (for now)
# could be extended to the multiclass case?
train_metrics = {}
if len(np.unique(outcome)) == 2:
    model = LogisticRegression(penalty="l2", solver="saga", C=best.C_[0])
    train_metrics["alpha"] = best.C_[0]
    # train_metrics["l1_ratio"] = best.l1_ratio_
else:
    model = Ridge(
        solver="auto",
        alpha=best.alpha_,
        fit_intercept=False,
    )
    train_metrics["alpha"] = best.alpha_

cv = RepeatedKFold(n_splits=5, n_repeats=10)

# train_metrics["l1_ratio"] = best.l1_ratio_
# print(params)
# model.set_params(**params)
# train ElasticNet on full train dataset, using feature extraction from NBS-Predict
# fitted = model.fit(X=train_features, y=np.ravel(train_outcome))
scores = cross_validate(
    model,
    train_features,
    train_outcome,
    groups=groups,
    cv=cv,
    return_estimator=True,
    return_train_score=True,
)
train_metrics["in_sample_test"] = np.mean(scores["test_score"])
train_metrics["in_sample_train"] = np.mean(scores["train_score"])

fitted = scores["estimator"][0]
y_pred = fitted.predict(X=train_features)
train_metrics["true_v_pred_corr"] = spearmanr(y_pred, train_outcome)

dat[f"{OUTCOME}_pred"] = y_pred
dat[f"{OUTCOME}_scaled"] = train_outcome

Ys = dat[[f"{OUTCOME}_pred", f"{OUTCOME}_scaled", "bc", "cycle_day"]]
Ys.to_csv(
    join(TRAIN_DSET, "derivatives", DERIV_NAME, f"{base_name}_actual-predicted.tsv"), sep="\t"
)

train_colors = ["#a08ad1", "#685690", "#3f2d69"]  # light  # medium  # dark
light_cmap = sns.color_palette("dark:#a08ad1")
dark_cmap = sns.color_palette("dark:#685690")

fig, ax = plt.subplots()
g = sns.scatterplot(
    x="cycle_day", y=f"{OUTCOME}_pred", style="bc", data=Ys, ax=ax, palette=dark_cmap
)
h = sns.scatterplot(
    x="cycle_day", y=f"{OUTCOME}_scaled", style="bc", data=Ys, ax=ax, palette=light_cmap
)
ax.legend(bbox_to_anchor=(1.0, 0.5))
fig.savefig(
    join(TRAIN_DSET, "derivatives", DERIV_NAME, f"{base_name}_actual-predicted.png"),
    dpi=400,
    bbox_inches="tight",
)

mse = mean_squared_error(train_outcome, y_pred)
train_metrics["mean squared error"] = mse
print("In-sample train score: ", train_metrics["in_sample_train"])
print("In-sample test score: ", train_metrics["in_sample_test"])
print("In-sample mean squared error: ", mse)
# print(np.mean(train_features))
with open(
    join(TRAIN_DSET, "derivatives", DERIV_NAME, f"{base_name}_fit-{today_str}.json"), "w"
) as fp:
    json.dump(train_metrics, fp)

# yoink the coefficients? for a more parsimonious figure?
# print(fitted.coef_.shape)
# print(fitted.coef_)
coeff_vec = np.zeros_like(filter)
j = 0
for i in range(0, filter.shape[0]):
    if filter[i] == True:
        # print(j)
        # print(fitted.coef_[0, j])
        coeff_vec[i] = fitted.coef_[0, j]
        j += 1
    else:
        pass

# print(coeff_vec)
print(coeff_vec)
coef_mat = io.undo_vectorize(coeff_vec, num_node=num_node)

coef_df = pd.DataFrame(coef_mat, columns=avg_df.columns, index=avg_df.index)
coef_df.to_csv(join(TRAIN_DSET, "derivatives", DERIV_NAME, f"{base_name}_betas-{today_str}.csv"))

fig, fig2, nimg = io.plot_edges(
    coef_mat,
    atlas_fname,
    threshold="computed",
    title=f"{OUTCOME} Coefficients",
    strength=True,
    cmap="seismic",
    node_size="strength",
)

fig.savefig(
    join(TRAIN_DSET, "derivatives", DERIV_NAME, f"{base_name}_betas-{today_str}.png"), dpi=400
)
fig2.savefig(
    join(TRAIN_DSET, "derivatives", DERIV_NAME, f"{base_name}_betas-strength-{today_str}.png"),
    dpi=400,
)
nib.save(
    nimg, join(TRAIN_DSET, "derivatives", DERIV_NAME, f"{base_name}_betas-strength-{today_str}")
)


layout = bids.BIDSLayout(TEST_DSET, derivatives=True)

test_df = io.read_corrmats(layout, task=TASK, deriv_name="IDConn", atlas=ATLAS, z_score=False)

keep = test_df[[OUTCOME, "adj"]].dropna().index
# print(keep)

test_df = test_df.loc[keep]

outcome_test = test_df[OUTCOME].values
# print(test_df)

# print(outcome_test)
matrices_test = np.vstack(test_df["adj"].dropna().values).reshape(
    (len(test_df["adj"].dropna().index), num_node, num_node)
)
edges_test = np.vstack(test_df["edge_vector"].dropna().values)[:, filter]

# NEED TO RESIDUALIZE IF CONFOUNDS IS NOT NONE
if CONFOUNDS is not None:
    confounds_test = test_df[CONFOUNDS].values

    # regress out the confounds from each edge and the outcome variable,
    # use the residuals for the rest of the algorithm
    # print(confounds.shape, outcome.shape)
    if len(np.unique(outcome_test)) <= 2:
        resid_edges = nbs.residualize(X=edges_test, confounds=confounds_test)
        test_outcome = outcome_test
    elif len(np.unique(outcome_test)) > 3:
        test_outcome, resid_edges = nbs.residualize(
            X=edges_test, y=outcome_test, confounds=confounds_test
        )
    test_features = resid_edges
else:
    test_features = edges_test
    test_outcome = outcome_test

# scale after residualizing omg
test_features = x_scaler.transform(test_features)
if len(np.unique(test_outcome)) <= 2:
    pass
else:
    test_outcome = y_scaler.transform(test_outcome.reshape(-1, 1))
# print(test_features.shape)
# if the model is a logistic regression, i.e. with a binary outcome
# then score is prediction accuracy
# if the model is a linear regression, i.e., with a continuous outcome
# then the score is R^2 (coefficient of determination)

# fit trained ElasticNet, initialized via warm_start
# prob in CV?
# fitted_test = fitted.fit(X=test_features, y=np.ravel(test_outcome))
# score = fitted_test.score(X=test_features, y=np.ravel(test_outcome))
test_metrics = {}

# cross_validate(model, )
y_pred = fitted.predict(X=test_features)
score = fitted.score(X=test_features, y=np.ravel(test_outcome))
if len(np.unique(test_outcome)) == 2:
    test_metrics["accuracy"] = score
else:
    test_metrics["coefficient of determination"] = score
corr = spearmanr(test_outcome, y_pred)
test_metrics["pred_v_actual_corr"] = corr
mse = mean_squared_error(test_outcome, y_pred)
test_metrics["mean squared error"] = mse
print("Out-of-sample prediction score:\t", score)
print("Out-of-sample mean squared error:\t", mse)
# print(np.mean(test_features))
# pred_outcome = fitted.predict(test_features)
test_df[f"{OUTCOME}_scaled"] = test_outcome
test_df[f"{OUTCOME}_pred"] = y_pred
Ys = test_df[[f"{OUTCOME}_scaled", f"{OUTCOME}_pred", "cycle_day", "bc"]]
Ys.to_csv(
    join(TEST_DSET, "derivatives", DERIV_NAME, f"{base_name}_actual-predicted.tsv"), sep="\t"
)

Ys["ppts"] = Ys.index.get_level_values(0)


light_colors = ["#33ACE3", "#EA6964", "#4AB62C"]  # Bubbles  # Blossom  # Buttercup
dark_colors = ["#1278a6", "#a11510", "#228208"]
light = ListedColormap(light_colors, name="light_powderpuff")
dark = ListedColormap(dark_colors, name="dark_powderpuff")
mpl.colormaps.register(cmap=light)
mpl.colormaps.register(cmap=dark)

fig, ax = plt.subplots()
g = sns.scatterplot(
    x="cycle_day",
    y=f"{OUTCOME}_pred",
    style="bc",
    data=Ys,
    hue="ppts",
    hue_order=["sub-Bubbles", "sub-Blossom", "sub-Buttercup"],
    ax=ax,
    palette="light_powderpuff",
)
h = sns.scatterplot(
    x="cycle_day",
    y=f"{OUTCOME}_scaled",
    style="bc",
    data=Ys,
    hue="ppts",
    hue_order=["sub-Bubbles", "sub-Blossom", "sub-Buttercup"],
    ax=ax,
    palette="dark_powderpuff",
)
ax.legend(bbox_to_anchor=(1.0, 0.5), loc="center left")
fig.savefig(
    join(TEST_DSET, "derivatives", DERIV_NAME, f"{base_name}_actual-predicted.png"),
    dpi=400,
    bbox_inches="tight",
)


# print(test_outcome, "\n", y_pred)
# print(pred_outcome)
if len(np.unique(test_outcome)) > 2:

    print(f"\nSpearman correlation between predicted and actual {OUTCOME}:\t", corr)
    test_metrics["spearman correlation"] = corr
with open(
    join(TEST_DSET, "derivatives", DERIV_NAME, f"{base_name}_fit-{today_str}.json"), "w"
) as fp:
    json.dump(test_metrics, fp)
np.savetxt(join(TEST_DSET, f"{base_name}_predicted-values_fit-{today_str}.txt"), y_pred)
