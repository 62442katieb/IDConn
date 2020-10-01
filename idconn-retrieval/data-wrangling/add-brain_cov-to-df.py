import numpy as np
import pandas as pd
import seaborn as sns
from os import makedirs
from os.path import join, exists

import datetime

sink_dir = "/Users/kbottenh/Dropbox/Projects/physics-retrieval/data/output"
fig_dir = "/Users/kbottenh/Dropbox/Projects/physics-retrieval/figures/"


data_dir = "/Users/kbottenh/Dropbox/Projects/physics-retrieval/data"
b_df = pd.read_csv(
    join(data_dir, "rescored", "physics_learning-nonbrain_OLS-BayesianImpute.csv"),
    index_col=0,
    header=0,
)

head_size = pd.read_csv(
    join(data_dir, "head-size_2019-05-29 15:19:53.287525.csv"), index_col=0, header=0
)
head_size["normalized head size"] = (
    head_size["average_head_size"] - np.mean(head_size["average_head_size"])
) / np.std(head_size["average_head_size"])

fd = pd.read_csv(
    join(data_dir, "avg-fd-per-condition-per-run_2019-05-29.csv"), index_col=0, header=0
)
fd["normalized fd"] = (fd["average fd"] - np.mean(fd["average fd"])) / np.std(
    fd["average fd"]
)
retr_fd = fd[fd["task"] == "retr"]
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

big_df = pd.concat([b_df, retr_phys_fd, retr_genr_fd, fci_phys_fd, fci_ctrl_fd], axis=1)

big_df.to_csv(join(data_dir, "rescored", "physics_learning-nonbrain_OLS+fd.csv"))
