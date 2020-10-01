# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
from os import makedirs
from os.path import join, exists
from nilearn.plotting import plot_connectome, plot_roi, find_parcellation_cut_coords
import bct
import datetime
from nilearn.mass_univariate import permuted_ols
from scipy.stats import pearsonr, spearmanr


# +


def jili_sidak_mc(data, alpha):
    import math
    import numpy as np

    mc_corrmat = data.corr()
    mc_corrmat.fillna(0, inplace=True)
    eigvals, eigvecs = np.linalg.eig(mc_corrmat)

    M_eff = 0
    for eigval in eigvals:
        if abs(eigval) >= 0:
            if abs(eigval) >= 1:
                M_eff += 1
            else:
                M_eff += abs(eigval) - math.floor(abs(eigval))
        else:
            M_eff += 0
    print("Number of effective comparisons: {0}".format(M_eff))

    # and now applying M_eff to the Sidak procedure
    sidak_p = 1 - (1 - alpha) ** (1 / M_eff)
    if sidak_p < 0.00001:
        print(
            "Critical value of {:.3f}".format(alpha),
            "becomes {:2e} after corrections".format(sidak_p),
        )
    else:
        print(
            "Critical value of {:.3f}".format(alpha),
            "becomes {:.6f} after corrections".format(sidak_p),
        )
    return sidak_p, M_eff


# +
subjects = [
    "101",
    "102",
    "103",
    "104",
    "106",
    "107",
    "108",
    "110",
    "212",
    "213",
    "214",
    "215",
    "216",
    "217",
    "218",
    "219",
    "320",
    "321",
    "322",
    "323",
    "324",
    "325",
    "327",
    "328",
    "329",
    "330",
    "331",
    "332",
    "333",
    "334",
    "335",
    "336",
    "337",
    "338",
    "339",
    "340",
    "341",
    "342",
    "343",
    "344",
    "345",
    "346",
    "347",
    "348",
    "349",
    "350",
    "451",
    "452",
    "453",
    "455",
    "456",
    "457",
    "458",
    "459",
    "460",
    "462",
    "463",
    "464",
    "465",
    "467",
    "468",
    "469",
    "470",
    "502",
    "503",
    "571",
    "572",
    "573",
    "574",
    "575",
    "577",
    "578",
    "579",
    "580",
    "581",
    "582",
    "584",
    "585",
    "586",
    "587",
    "588",
    "589",
    "590",
    "591",
    "592",
    "593",
    "594",
    "595",
    "596",
    "597",
    "598",
    "604",
    "605",
    "606",
    "607",
    "608",
    "609",
    "610",
    "611",
    "612",
    "613",
    "614",
    "615",
    "616",
    "617",
    "618",
    "619",
    "620",
    "621",
    "622",
    "623",
    "624",
    "625",
    "626",
    "627",
    "628",
    "629",
    "630",
    "631",
    "633",
    "634",
]
# subjects = ['101', '102']
kappa_upper = 0.21
kappa_lower = 0.31

sink_dir = "/Users/Katie/Dropbox/Projects/physics-retrieval/data/output"

shen = "/home/kbott006/physics-retrieval/shen2015_2mm_268_parcellation.nii.gz"
craddock = "/home/kbott006/physics-retrieval/craddock2012_tcorr05_2level_270_2mm.nii.gz"
masks = ["shen2015", "craddock2012"]

tasks = {
    "reas": [{"conditions": ["Reasoning", "Baseline"]}, {"runs": [0, 1]}],
    "retr": [{"conditions": ["Physics", "General"]}, {"runs": [0, 1]}],
    "fci": [{"conditions": ["Physics", "NonPhysics"]}, {"runs": [0, 1, 2]}],
}

sessions = [0, 1]
sesh = ["pre", "post"]
conds = ["high-level", "lower-level"]

lab_notebook_dir = "/home/kbott006/lab_notebook/"
index = pd.MultiIndex.from_product(
    [subjects, sessions, tasks, conds, masks],
    names=["subject", "session", "task", "condition", "mask"],
)
lab_notebook = pd.DataFrame(index=index, columns=["start", "end", "errors"])

index = pd.MultiIndex.from_product(
    [subjects, sessions, tasks, conds, masks],
    names=["subject", "session", "task", "condition", "mask"],
)
# -

shen_df = pd.read_csv(
    join(sink_dir, "task-shen-triplenetwork.csv"), index_col=0, header=0
)
shen_df.rename(
    {"Unnamed: 1": "session", "Unnamed: 2": "task", "Unnamed: 3": "condition"},
    axis=1,
    inplace=True,
)
shen_rest_df = pd.read_csv(
    join(sink_dir, "rest-shen-triplenetwork.csv"), index_col=0, header=0
)
shen_rest_df.rename({"Unnamed: 1": "session"}, axis=1, inplace=True)

shen_df.head()

rest_pre = shen_rest_df[shen_rest_df["session"] == 0].dropna()
rest_post = shen_rest_df[shen_rest_df["session"] == 1].dropna()

# +
data_dir = "/Users/Katie/Dropbox/Projects/physics-retrieval/data"
b_df = pd.read_csv(
    join(data_dir, "rescored", "non-brain-data.csv"), index_col=0, header=0
)

df_f = b_df[b_df["Sex"] == "F"]
df_f = df_f.drop("Sex", axis=1)
df_m = b_df[b_df["Sex"] == "M"]
df_m = df_m.drop("Sex", axis=1)

df_f["const"] = 1
df_m["const"] = 1
# -

rest_fd = pd.read_csv(
    "/Users/Katie/Dropbox/Projects/physics-retrieval/data/avg-fd-per-run-rest_2019-05-31.csv",
    index_col=0,
    header=0,
)
rest_fd["normalized fd"] = (
    rest_fd["average fd"] - np.mean(rest_fd["average fd"])
) / np.std(rest_fd["average fd"])

df_pivot = rest_fd.reset_index()
rest_fd = df_pivot.pivot(index="subject", columns="session", values="normalized fd")
rest_fd.rename({"pre": "pre rest fd", "post": "post rest fd"}, axis=1, inplace=True)

# +
fd = pd.read_csv(
    join(
        "/Users/Katie/Dropbox/Projects/physics-retrieval/data/avg-fd-per-condition-per-run_2019-05-29.csv"
    ),
    index_col=0,
    header=0,
)
fd["normalized fd"] = (fd["average fd"] - np.mean(fd["average fd"])) / np.std(
    fd["average fd"]
)
retr_fd = fd[fd["task"] == "retr"]
reas_fd = fd[fd["task"] == "reas"]
fci_fd = fd[fd["task"] == "fci"]

df_pivot = retr_fd[retr_fd["condition"] == "high-level"].reset_index()
retr_phys_fd = df_pivot.pivot(index="subject", columns="session", values="average fd")
retr_phys_fd.rename(
    {"pre": "pre phys retr fd", "post": "post phys retr fd"}, axis=1, inplace=True
)

df_pivot = retr_fd[retr_fd["condition"] == "lower-level"].reset_index()
retr_genr_fd = df_pivot.pivot(index="subject", columns="session", values="average fd")
retr_genr_fd.rename(
    {"pre": "pre gen retr fd", "post": "post gen retr fd"}, axis=1, inplace=True
)

df_pivot = reas_fd[reas_fd["condition"] == "high-level"].reset_index()
reas_inf_fd = df_pivot.pivot(index="subject", columns="session", values="average fd")
reas_inf_fd.rename(
    {"pre": "pre infr reas fd", "post": "post infr reas fd"}, axis=1, inplace=True
)

df_pivot = reas_fd[reas_fd["condition"] == "lower-level"].reset_index()
reas_base_fd = df_pivot.pivot(index="subject", columns="session", values="average fd")
reas_base_fd.rename(
    {"pre": "pre base reas fd", "post": "post base reas fd"}, axis=1, inplace=True
)

df_pivot = fci_fd[fci_fd["condition"] == "high-level"].reset_index()
fci_phys_fd = df_pivot.pivot(index="subject", columns="session", values="average fd")
fci_phys_fd.rename(
    {"pre": "pre phys fci fd", "post": "post phys fci fd"}, axis=1, inplace=True
)

df_pivot = fci_fd[fci_fd["condition"] == "lower-level"].reset_index()
fci_ctrl_fd = df_pivot.pivot(index="subject", columns="session", values="average fd")
fci_ctrl_fd.rename(
    {"pre": "pre ctrl fci fd", "post": "post ctrl fci fd"}, axis=1, inplace=True
)
# -

iqs = ["VCI", "WMI", "PRI", "PSI", "FSIQ"]

big_df = pd.concat(
    [
        b_df,
        retr_phys_fd,
        retr_genr_fd,
        fci_phys_fd,
        fci_ctrl_fd,
        reas_base_fd,
        reas_inf_fd,
        rest_fd,
    ],
    axis=1,
)

fci_shen = shen_df[shen_df["task"] == "fci"]
fci_shen_pre = fci_shen[fci_shen["session"] == 0]
fci_shen_pre_phys = fci_shen_pre[fci_shen_pre["condition"] == "high-level"]
fci_shen_pre_ctrl = fci_shen_pre[fci_shen_pre["condition"] == "lower-level"]
fci_shen_post = fci_shen[fci_shen["session"] == 1]
fci_shen_post_phys = fci_shen_post[fci_shen_post["condition"] == "high-level"]
fci_shen_post_ctrl = fci_shen_post[fci_shen_post["condition"] == "lower-level"]

retr_shen = shen_df[shen_df["task"] == "retr"]
retr_shen_pre = retr_shen[retr_shen["session"] == 0]
retr_shen_pre_phys = retr_shen_pre[retr_shen_pre["condition"] == "high-level"]
retr_shen_pre_ctrl = retr_shen_pre[retr_shen_pre["condition"] == "lower-level"]
retr_shen_post = retr_shen[retr_shen["session"] == 1]
retr_shen_post_phys = retr_shen_post[retr_shen_post["condition"] == "high-level"]
retr_shen_post_ctrl = retr_shen_post[retr_shen_post["condition"] == "lower-level"]

reas_shen = shen_df[shen_df["task"] == "reas"]
reas_shen_pre = reas_shen[reas_shen["session"] == 0]
reas_shen_pre_infr = reas_shen_pre[reas_shen_pre["condition"] == "high-level"]
reas_shen_pre_ctrl = reas_shen_pre[reas_shen_pre["condition"] == "lower-level"]
reas_shen_post = reas_shen[reas_shen["session"] == 1]
reas_shen_post_infr = reas_shen_post[reas_shen_post["condition"] == "high-level"]
reas_shen_post_ctrl = reas_shen_post[reas_shen_post["condition"] == "lower-level"]

# +
fci_shen_pre_phys.drop(["session", "task", "condition"], axis=1, inplace=True)
fci_shen_post_phys.drop(["session", "task", "condition"], axis=1, inplace=True)

fci_shen_pre_ctrl.drop(["session", "task", "condition"], axis=1, inplace=True)
fci_shen_post_ctrl.drop(["session", "task", "condition"], axis=1, inplace=True)

retr_shen_pre_phys.drop(["session", "task", "condition"], axis=1, inplace=True)
retr_shen_post_phys.drop(["session", "task", "condition"], axis=1, inplace=True)

retr_shen_pre_ctrl.drop(["session", "task", "condition"], axis=1, inplace=True)
retr_shen_post_ctrl.drop(["session", "task", "condition"], axis=1, inplace=True)

reas_shen_pre_ctrl.drop(["session", "task", "condition"], axis=1, inplace=True)
reas_shen_pre_infr.drop(["session", "task", "condition"], axis=1, inplace=True)

rest_pre.drop(["session"], axis=1, inplace=True)
rest_post.drop(["session"], axis=1, inplace=True)
# -

# from all ppts with data, remove the female participants, then remove the participants with missing brain data
male_rest_index = list(
    set(rest_pre.index.values) - set(big_df[big_df["Sex"] == "F"].index.values)
)
feml_rest_index = list(
    set(rest_pre.index.values) - set(big_df[big_df["Sex"] == "M"].index.values)
)

# +
f_rest_pre = rest_pre.drop(male_rest_index, axis=0)
m_rest_pre = rest_pre.drop(feml_rest_index, axis=0)

f_rest_post = rest_pre.drop(male_rest_index, axis=0)
m_rest_post = rest_pre.drop(feml_rest_index, axis=0)
# -

m_rest_post.index.shape

for column in rest_pre.columns:
    num = np.nonzero(f_rest_pre[column].values)[0].shape
    if num[0] <= 5:
        f_rest_pre.drop(column, axis=1, inplace=True)
    num = np.nonzero(m_rest_pre[column].values)[0].shape
    if num[0] <= 5:
        m_rest_pre.drop(column, axis=1, inplace=True)

for column in rest_post.columns:
    num = np.nonzero(f_rest_post[column].values)[0].shape
    if num[0] <= 5:
        f_rest_post.drop(column, axis=1, inplace=True)
    num = np.nonzero(m_rest_post[column].values)[0].shape
    if num[0] <= 5:
        m_rest_post.drop(column, axis=1, inplace=True)

conns = list(set(f_rest_pre.columns))

# +
sig = {}

for iq in iqs:
    # RESTING STATE HOLLA (NO BOYS ALLOWED)
    all_data = pd.concat([big_df, f_rest_pre], axis=1)
    all_data.dropna(how="any", inplace=True)
    conns = list(set(f_rest_pre.columns))
    # drop_behav = set(big_df_f.index.values) - set(f_rest_pre.dropna(how='all').index.values)
    p, t, _ = permuted_ols(
        all_data["{0}1".format(iq)].values,
        all_data[conns].values,
        all_data[["Age", "Mod", "Strt.Level", "pre rest fd"]].values,
    )
    sig["Female students, pre: {0}, rest".format(iq)] = np.max(p[0])

    all_data = pd.concat([big_df, f_rest_post], axis=1)
    all_data.dropna(how="any", inplace=True)
    conns = list(set(f_rest_post.columns))

    p, t, _ = permuted_ols(
        all_data["{0}2".format(iq)].values,
        all_data[conns].values,
        all_data[["Age", "Mod", "Strt.Level", "post rest fd"]].values,
    )
    sig["Female students, post: {0}, rest".format(iq)] = np.max(p[0])

    p, t, _ = permuted_ols(
        all_data["delta{0}".format(iq)].values,
        all_data[conns].values,
        all_data[["Age", "Mod", "Strt.Level", "post rest fd"]].values,
    )
    sig["Female students, delta: {0}, rest".format(iq)] = np.max(p[0])

    # AND NOW FOR THE DUDES
    all_data = pd.concat([big_df, m_rest_pre], axis=1)
    all_data.dropna(how="any", inplace=True)
    conns = list(set(m_rest_pre.columns))
    # drop_behav = set(big_df_f.index.values) - set(f_rest_pre.dropna(how='all').index.values)
    p, t, _ = permuted_ols(
        all_data["{0}1".format(iq)].values,
        all_data[conns].values,
        all_data[["Age", "Mod", "Strt.Level", "pre rest fd"]].values,
    )
    sig["Male students, pre: {0}, rest".format(iq)] = np.max(p[0])

    all_data = pd.concat([big_df, m_rest_post], axis=1)
    all_data.dropna(how="any", inplace=True)
    conns = list(set(m_rest_post.columns))

    p, t, _ = permuted_ols(
        all_data["{0}2".format(iq)].values,
        all_data[conns].values,
        all_data[["Age", "Mod", "Strt.Level", "post rest fd"]].values,
    )
    sig["Male students, post: {0}, rest".format(iq)] = np.max(p[0])

    p, t, _ = permuted_ols(
        all_data["delta{0}".format(iq)].values,
        all_data[conns].values,
        all_data[["Age", "Mod", "Strt.Level", "post rest fd"]].values,
    )
    sig["Male students, delta: {0}, rest".format(iq)] = np.max(p[0])

# +
f_conns = retr_shen_pre_phys.drop(male_rest_index, axis=0)
m_conns = retr_shen_pre_phys.drop(feml_rest_index, axis=0)

for column in fci_shen_pre_phys.columns:
    num = np.nonzero(f_conns[column].values)[0].shape
    if num[0] <= 10:
        f_conns.drop(column, axis=1, inplace=True)
    num = np.nonzero(m_conns[column].values)[0].shape
    if num[0] <= 5:
        m_conns.drop(column, axis=1, inplace=True)

# +
all_data = pd.concat([big_df, f_conns], axis=1)
all_data.dropna(how="any", axis=0, inplace=True)
conns = list(set(f_conns.columns))
for iq in iqs:
    p, t, _ = permuted_ols(
        all_data["{0}1".format(iq)].values,
        all_data[conns].values,
        all_data[["Age", "Mod", "Strt.Level", "pre phys retr fd"]].values,
    )
    sig["Female students, pre: {0}, phys retr".format(iq)] = np.max(p[0])

all_data = pd.concat([big_df, m_conns], axis=1)
all_data.dropna(how="any", axis=0, inplace=True)
for iq in iqs:
    p, t, _ = permuted_ols(
        all_data["{0}1".format(iq)].values,
        all_data[conns].values,
        all_data[["Age", "Mod", "Strt.Level", "pre phys retr fd"]].values,
    )
    sig["Male students, pre: {0}, phys retr".format(iq)] = np.max(p[0])

# +
m_conns = fci_shen_pre_phys.drop(feml_rest_index, axis=0)
all_data = pd.concat([big_df, f_conns], axis=1)
all_data.dropna(how="any", axis=0, inplace=True)
conns = list(set(f_conns.columns))
for column in all_data[conns]:
    if m_conns[column].std() < 0.1:
        m_conns.drop(column, axis=1, inplace=True)

for iq in iqs:
    p, t, _ = permuted_ols(
        all_data["{0}1".format(iq)].values,
        all_data[conns].values,
        all_data[["Age", "Mod", "Strt.Level", "pre phys fci fd"]].values,
    )
    sig["Male students, pre: {0}, phys fci".format(iq)] = np.max(p[0])

# +
f_conns = fci_shen_pre_phys.drop(male_rest_index, axis=0)
all_data = pd.concat([big_df, f_conns], axis=1)
all_data.dropna(how="any", axis=0, inplace=True)
conns = list(set(f_conns.columns))

for column in all_data[conns]:
    if all_data[column].std() < 0.05:
        all_data.drop(column, axis=1, inplace=True)
        conns = list(set(conns) - set([column]))
# -

for iq in iqs:
    p, t, _ = permuted_ols(
        all_data["{0}1".format(iq)].values,
        all_data[conns].values,
        all_data[["Age", "Mod", "Strt.Level", "pre phys fci fd"]].values,
    )
    sig["Female students, pre: {0}, phys fci".format(iq)] = np.max(p[0])


# +
f_conns = fci_shen_post_phys.drop(male_rest_index, axis=0)
all_data = pd.concat([big_df, f_conns], axis=1)
all_data.dropna(how="any", axis=0, inplace=True)
conns = list(set(f_conns.columns))

for column in all_data[conns]:
    if all_data[column].std() < 0.05:
        all_data.drop(column, axis=1, inplace=True)
        conns = list(set(conns) - set([column]))

for iq in iqs:
    p, t, _ = permuted_ols(
        all_data["{0}2".format(iq)].values,
        all_data[conns].values,
        all_data[["Age", "Mod", "Strt.Level", "post phys fci fd"]].values,
    )
    sig["Female students, post: {0}, phys fci".format(iq)] = np.max(p[0])
    p, t, _ = permuted_ols(
        all_data["delta{0}".format(iq)].values,
        all_data[conns].values,
        all_data[["Age", "Mod", "Strt.Level", "post phys fci fd"]].values,
    )
    sig["Female students, delta: {0}, phys fci".format(iq)] = np.max(p[0])

# +
m_conns = fci_shen_post_phys.drop(feml_rest_index, axis=0)

for column in fci_shen_post_phys.columns:
    num = np.nonzero(f_conns[column].values)[0].shape
    if num[0] <= 10:
        f_conns.drop(column, axis=1, inplace=True)
    num = np.nonzero(m_conns[column].values)[0].shape
    if num[0] <= 5:
        m_conns.drop(column, axis=1, inplace=True)

# +
all_data = pd.concat([big_df, f_conns], axis=1)
all_data.dropna(how="any", axis=0, inplace=True)
conns = list(set(f_conns.columns))

all_data = pd.concat([big_df, m_conns], axis=1)
all_data.dropna(how="any", axis=0, inplace=True)
for iq in iqs:
    p, t, _ = permuted_ols(
        all_data["{0}2".format(iq)].values,
        all_data[conns].values,
        all_data[["Age", "Mod", "Strt.Level", "post phys fci fd"]].values,
    )
    sig["Male students, post: {0}, phys fci".format(iq)] = np.max(p[0])

    p, t, _ = permuted_ols(
        all_data["delta{0}".format(iq)].values,
        all_data[conns].values,
        all_data[["Age", "Mod", "Strt.Level", "post phys fci fd"]].values,
    )
    sig["Male students, delta: {0}, phys fci".format(iq)] = np.max(p[0])

# +
f_conns = retr_shen_post_phys.drop(male_rest_index, axis=0)
m_conns = retr_shen_post_phys.drop(feml_rest_index, axis=0)

for column in retr_shen_post_phys.columns:
    num = np.nonzero(f_conns[column].values)[0].shape
    if num[0] <= 10:
        f_conns.drop(column, axis=1, inplace=True)
    num = np.nonzero(m_conns[column].values)[0].shape
    if num[0] <= 5:
        m_conns.drop(column, axis=1, inplace=True)

# +
all_data = pd.concat([big_df, f_conns], axis=1)
all_data.dropna(how="any", axis=0, inplace=True)
conns = list(set(f_conns.columns))
for iq in iqs:
    p, t, _ = permuted_ols(
        all_data["{0}2".format(iq)].values,
        all_data[conns].values,
        all_data[["Age", "Mod", "Strt.Level", "post phys retr fd"]].values,
    )
    sig["Female students, post: {0}, phys retr".format(iq)] = np.max(p[0])

    p, t, _ = permuted_ols(
        all_data["delta{0}".format(iq)].values,
        all_data[conns].values,
        all_data[["Age", "Mod", "Strt.Level", "post phys retr fd"]].values,
    )
    sig["Female students, delta: {0}, phys retr".format(iq)] = np.max(p[0])

all_data = pd.concat([big_df, m_conns], axis=1)
all_data.dropna(how="any", axis=0, inplace=True)
for iq in iqs:
    p, t, _ = permuted_ols(
        all_data["{0}2".format(iq)].values,
        all_data[conns].values,
        all_data[["Age", "Mod", "Strt.Level", "post phys retr fd"]].values,
    )
    sig["Male students, post: {0}, phys retr".format(iq)] = np.max(p[0])

    p, t, _ = permuted_ols(
        all_data["delta{0}".format(iq)].values,
        all_data[conns].values,
        all_data[["Age", "Mod", "Strt.Level", "post phys retr fd"]].values,
    )
    sig["Male students, delta: {0}, phys retr".format(iq)] = np.max(p[0])
# -

for key in sig.keys():
    if sig[key] >= 1:
        print(key, sig[key])

coordinates = find_parcellation_cut_coords(
    labels_img="/Users/Katie/Dropbox/Data/templates/shen2015/shen_triplenetwork.nii.gz"
)

# +
sig_f = {"PRI": ["pre rest", f_rest_pre]}

for key in sig_f.keys():
    f_conns = sig_f[key][1]
    iq = key
    task = sig_f[key][0]
    print(iq, task)
    all_data = pd.concat([big_df, f_conns], axis=1)
    all_data.dropna(how="any", axis=0, inplace=True)

    conns = list(set(f_conns.columns))

    for column in all_data[conns]:
        if all_data[column].std() < 0.05:
            all_data.drop(column, axis=1, inplace=True)
            conns = list(set(conns) - set([column]))

    p, t, _ = permuted_ols(
        all_data["{0}1".format(iq)].values,
        all_data[conns].values,
        all_data[["Age", "Mod", "Strt.Level", "{0} fd".format(task)]].values,
    )
    triplenetwork_sig = pd.DataFrame(index=["p", "t"], columns=conns)
    triplenetwork_sig.at["p"] = p
    triplenetwork_sig.at["t"] = t
    index = []
    for column in triplenetwork_sig.columns:
        index.append(column.split("-")[0])
    regions = list(set(index))
    sig_prewmi_fci_p = pd.DataFrame(index=regions, columns=regions)
    sig_prewmi_fci_t = pd.DataFrame(index=regions, columns=regions)
    for region1 in regions:
        for region2 in regions:
            try:
                sig_prewmi_fci_p.at[region1, region2] = triplenetwork_sig.loc["p"][
                    "{0}-{1}".format(region1, region2)
                ]
                sig_prewmi_fci_t.at[region1, region2] = triplenetwork_sig.loc["t"][
                    "{0}-{1}".format(region1, region2)
                ]
            except Exception as e:
                pass
    for i in sig_prewmi_fci_p.index:
        if len(i) == 5:
            sig_prewmi_fci_t.rename({"{0}".format(i): "0{0}".format(i)}, inplace=True)
            sig_prewmi_fci_p.rename({"{0}".format(i): "0{0}".format(i)}, inplace=True)
        if len(i) == 4:
            sig_prewmi_fci_t.rename({"{0}".format(i): "00{0}".format(i)}, inplace=True)
            sig_prewmi_fci_p.rename({"{0}".format(i): "00{0}".format(i)}, inplace=True)
    for i in sig_prewmi_fci_p.columns:
        if len(i) == 5:
            sig_prewmi_fci_t.rename(
                {"{0}".format(i): "0{0}".format(i)}, axis=1, inplace=True
            )
            sig_prewmi_fci_p.rename(
                {"{0}".format(i): "0{0}".format(i)}, axis=1, inplace=True
            )
        if len(i) == 4:
            sig_prewmi_fci_t.rename(
                {"{0}".format(i): "00{0}".format(i)}, axis=1, inplace=True
            )
            sig_prewmi_fci_p.rename(
                {"{0}".format(i): "00{0}".format(i)}, axis=1, inplace=True
            )
    sig_prewmi_fci_p.sort_index(inplace=True)
    sig_prewmi_fci_p.sort_index(axis=1, inplace=True)
    sig_prewmi_fci_t.sort_index(inplace=True)
    sig_prewmi_fci_t.sort_index(axis=1, inplace=True)
    print("Significant connections:")
    for i in sig_prewmi_fci_p.index:
        for j in sig_prewmi_fci_p.columns:
            if sig_prewmi_fci_p.at[i, j] >= 1:
                print(i, j, sig_prewmi_fci_t.at[i, j])
    sig_prewmi_fci_p.fillna(0, inplace=True)
    sig_prewmi_fci_t.fillna(0, inplace=True)
    q = plot_connectome(sig_prewmi_fci_p.values, coordinates, edge_threshold=1.0)
    sig_prewmi_fci_p.to_csv(join(sink_dir, "f-{0}-{1}_conn_pvals.csv".format(task, iq)))
    sig_prewmi_fci_t.to_csv(join(sink_dir, "f-{0}-{1}_conn-tvals.csv".format(task, iq)))
    q.savefig(
        "/Users/Katie/Dropbox/Projects/physics-retrieval/figures/f-{0}-{1}_conn_sig.png".format(
            task, iq
        ),
        dpi=300,
    )

# +
sig_f = {
    "WMI": ["pre phys fci", fci_shen_pre_phys],
    "FSIQ": ["pre phys fci", fci_shen_pre_phys],
}

pvals = {}
tvals = {}

for key in sig_f.keys():
    df_conn = sig_f[key][1]
    iq = key
    task = sig_f[key][0]
    print(iq, task)
    f_conns = df_conn.drop(male_rest_index, axis=0)
    all_data = pd.concat([big_df, f_conns], axis=1)
    all_data.dropna(how="any", axis=0, inplace=True)

    conns = list(set(f_conns.columns))

    for column in all_data[conns]:
        if all_data[column].std() < 0.05:
            all_data.drop(column, axis=1, inplace=True)
            conns = list(set(conns) - set([column]))

    p, t, _ = permuted_ols(
        all_data["{0}1".format(iq)].values,
        all_data[conns].values,
        all_data[["Age", "Mod", "Strt.Level", "{0} fd".format(task)]].values,
    )
    triplenetwork_sig = pd.DataFrame(index=["p", "t"], columns=conns)
    triplenetwork_sig.at["p"] = p
    triplenetwork_sig.at["t"] = t
    index = []
    for column in triplenetwork_sig.columns:
        index.append(column.split("-")[0])
    regions = list(set(index))
    sig_prewmi_fci_p = pd.DataFrame(index=regions, columns=regions)
    sig_prewmi_fci_t = pd.DataFrame(index=regions, columns=regions)
    for region1 in regions:
        for region2 in regions:
            try:
                sig_prewmi_fci_p.at[region1, region2] = triplenetwork_sig.loc["p"][
                    "{0}-{1}".format(region1, region2)
                ]
                sig_prewmi_fci_t.at[region1, region2] = triplenetwork_sig.loc["t"][
                    "{0}-{1}".format(region1, region2)
                ]
            except Exception as e:
                pass
    for i in sig_prewmi_fci_p.index:
        if len(i) == 5:
            sig_prewmi_fci_t.rename({"{0}".format(i): "0{0}".format(i)}, inplace=True)
            sig_prewmi_fci_p.rename({"{0}".format(i): "0{0}".format(i)}, inplace=True)
        if len(i) == 4:
            sig_prewmi_fci_t.rename({"{0}".format(i): "00{0}".format(i)}, inplace=True)
            sig_prewmi_fci_p.rename({"{0}".format(i): "00{0}".format(i)}, inplace=True)
    for i in sig_prewmi_fci_p.columns:
        if len(i) == 5:
            sig_prewmi_fci_t.rename(
                {"{0}".format(i): "0{0}".format(i)}, axis=1, inplace=True
            )
            sig_prewmi_fci_p.rename(
                {"{0}".format(i): "0{0}".format(i)}, axis=1, inplace=True
            )
        if len(i) == 4:
            sig_prewmi_fci_t.rename(
                {"{0}".format(i): "00{0}".format(i)}, axis=1, inplace=True
            )
            sig_prewmi_fci_p.rename(
                {"{0}".format(i): "00{0}".format(i)}, axis=1, inplace=True
            )
    sig_prewmi_fci_p.sort_index(inplace=True)
    sig_prewmi_fci_p.sort_index(axis=1, inplace=True)
    sig_prewmi_fci_t.sort_index(inplace=True)
    sig_prewmi_fci_t.sort_index(axis=1, inplace=True)
    print("Significant connections:")
    for i in sig_prewmi_fci_p.index:
        for j in sig_prewmi_fci_p.columns:
            if sig_prewmi_fci_p.at[i, j] >= 1:
                print(i, j, sig_prewmi_fci_t.at[i, j])
    sig_prewmi_fci_p.fillna(0, inplace=True)
    sig_prewmi_fci_t.fillna(0, inplace=True)
    q = plot_connectome(sig_prewmi_fci_p.values, coordinates, edge_threshold=1.0)
    sig_prewmi_fci_p.to_csv(join(sink_dir, "f-{0}-{1}_conn_pvals.csv".format(task, iq)))
    sig_prewmi_fci_t.to_csv(join(sink_dir, "f-{0}-{1}_conn-tvals.csv".format(task, iq)))
    q.savefig(
        "/Users/Katie/Dropbox/Projects/physics-retrieval/figures/f-{0}-{1}_conn_sig.png".format(
            task, iq
        ),
        dpi=300,
    )

# +
sig_m = {"PSI-delta": ["post phys retr", retr_shen_post_phys]}

pvals = {}
tvals = {}

for key in sig_m.keys():
    df_conn = sig_m[key][1]
    iq = key.split("-")[0]
    session = key.split("-")[1]
    task = sig_m[key][0]
    print(iq, session, task)
    m_conns = df_conn.drop(feml_rest_index, axis=0)
    all_data = pd.concat([big_df, m_conns], axis=1)
    all_data.dropna(how="any", axis=0, inplace=True)

    conns = list(set(m_conns.columns))

    for column in all_data[conns]:
        if all_data[column].std() < 0.05:
            all_data.drop(column, axis=1, inplace=True)
            conns = list(set(conns) - set([column]))
    if session == "delta":
        p, t, _ = permuted_ols(
            all_data["{0}{1}".format(session, iq)].values,
            all_data[conns].values,
            all_data[["Age", "Mod", "Strt.Level", "{0} fd".format(task)]].values,
        )
    else:
        p, t, _ = permuted_ols(
            all_data["{0}{1}".format(iq, session)].values,
            all_data[conns].values,
            all_data[["Age", "Mod", "Strt.Level", "{0} fd".format(task)]].values,
        )
    triplenetwork_sig = pd.DataFrame(index=["p", "t"], columns=conns)
    triplenetwork_sig.at["p"] = p
    triplenetwork_sig.at["t"] = t
    index = []
    for column in conns:
        index.append(column.split("-")[0])
    regions = list(set(index))
    sig_prewmi_fci_p = pd.DataFrame(index=regions, columns=regions)
    sig_prewmi_fci_t = pd.DataFrame(index=regions, columns=regions)
    for region1 in regions:
        for region2 in regions:
            try:
                sig_prewmi_fci_p.at[region1, region2] = triplenetwork_sig.loc["p"][
                    "{0}-{1}".format(region1, region2)
                ]
                sig_prewmi_fci_t.at[region1, region2] = triplenetwork_sig.loc["t"][
                    "{0}-{1}".format(region1, region2)
                ]
            except Exception as e:
                pass
    for i in sig_prewmi_fci_p.index:
        if len(i) == 5:
            sig_prewmi_fci_t.rename({"{0}".format(i): "0{0}".format(i)}, inplace=True)
            sig_prewmi_fci_p.rename({"{0}".format(i): "0{0}".format(i)}, inplace=True)
        if len(i) == 4:
            sig_prewmi_fci_t.rename({"{0}".format(i): "00{0}".format(i)}, inplace=True)
            sig_prewmi_fci_p.rename({"{0}".format(i): "00{0}".format(i)}, inplace=True)
    for i in sig_prewmi_fci_p.columns:
        if len(i) == 5:
            sig_prewmi_fci_t.rename(
                {"{0}".format(i): "0{0}".format(i)}, axis=1, inplace=True
            )
            sig_prewmi_fci_p.rename(
                {"{0}".format(i): "0{0}".format(i)}, axis=1, inplace=True
            )
        if len(i) == 4:
            sig_prewmi_fci_t.rename(
                {"{0}".format(i): "00{0}".format(i)}, axis=1, inplace=True
            )
            sig_prewmi_fci_p.rename(
                {"{0}".format(i): "00{0}".format(i)}, axis=1, inplace=True
            )
    sig_prewmi_fci_p.sort_index(inplace=True)
    sig_prewmi_fci_p.sort_index(axis=1, inplace=True)
    sig_prewmi_fci_t.sort_index(inplace=True)
    sig_prewmi_fci_t.sort_index(axis=1, inplace=True)
    print("Significant connections:")
    for i in sig_prewmi_fci_p.index:
        for j in sig_prewmi_fci_p.columns:
            if sig_prewmi_fci_p.at[i, j] >= 1:
                print(i, j, sig_prewmi_fci_t.at[i, j])
    sig_prewmi_fci_p.fillna(0, inplace=True)
    sig_prewmi_fci_t.fillna(0, inplace=True)
    sig_prewmi_fci_p.to_csv(
        join(sink_dir, "m-{0}-{1}{2}_conn_pvals.csv".format(task, iq, session))
    )
    sig_prewmi_fci_t.to_csv(
        join(sink_dir, "m-{0}-{1}{2}_conn-tvals.csv".format(task, iq, sessions))
    )
    q = plot_connectome(
        sig_prewmi_fci_p.values, coordinates, edge_threshold=1.0, edge_cmap="Blues"
    )
    q.savefig(
        "/Users/Katie/Dropbox/Projects/physics-retrieval/figures/m-{0}-{1}{2}_conn_sig.png".format(
            task, iq, session
        ),
        dpi=300,
    )
# -
