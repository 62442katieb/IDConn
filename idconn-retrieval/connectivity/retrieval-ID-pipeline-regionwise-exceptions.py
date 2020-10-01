from __future__ import division
from os.path import join, basename, exists
from os import makedirs
from glob import glob

from nilearn import input_data, datasets, plotting
from nilearn.image import concat_imgs
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from scipy.stats import pearsonr

import bct
import json
import numpy as np
import pandas as pd

labels = [
    "limbic",
    "limbic",
    "orbitofrontal",
    "orbitofrontal",
    "basal ganglia",
    "salience",
    "salience",
    "salience",
    "hunger",
    "hunger",
    "hunger",
    "hunger",
    "hunger",
    "hunger",
    "hunger",
    "motor learning",
    "frontoparietal",
    "frontoparietal",
    "frontoparietal",
    "hand",
    "hand",
    "hand",
    "motor execution",
    "motor execution",
    "higher order visual",
    "higher order visual",
    "lateral visual",
    "lateral visual",
    "medial visual",
    "default mode",
    "default mode",
    "default mode",
    "default mode",
    "default mode",
    " cerebellum",
    "right central executive",
    "right central executive",
    "right central executive",
    "right central executive",
    "right central executive",
    "auditory",
    "auditory",
    "mouth",
    "mouth",
    "left central executive",
    "left central executive",
    "left central executive",
]


# only two problem subjects
subjects = ["321", "618"]


# data_dir = '/home/data/nbc/physics-learning/data/pre-processed'
data_dir = "/home/data/nbc/physics-learning/retrieval-graphtheory/output"
sink_dir = "/home/kbott006/physics-retrieval/output"

runs = [0, 1]
connectivity_metric = "correlation"
conditions = ["phy", "gen"]
thresh_range = np.arange(0.1, 1, 0.1)
highpass = 1 / 55.0

correlation_measure = ConnectivityMeasure(kind=connectivity_metric)


# In[ ]:


# gen_timing = np.genfromtxt('/home/data/nbc/physics-learning/physics-learning/RETRconditionGeneralSess1.txt',
#                           delimiter='\t')
gen_timing = np.genfromtxt(
    "/home/data/nbc/physics-learning/retrieval-graphtheory/RETRconditionGeneralSess1.txt",
    delimiter="\t",
    dtype=int,
)

gen_timing = (gen_timing / 2) - 1
gen_timing = gen_timing[:, 0:2]

# phy_timing = np.genfromtxt('/home/data/nbc/physics-learning/physics-learning/RETRconditionPhysicsSess1.txt',
#                           delimiter='\t')
phy_timing = np.genfromtxt(
    "/home/data/nbc/physics-learning/retrieval-graphtheory/RETRconditionPhysicsSess1.txt",
    delimiter="\t",
)
phy_timing = (phy_timing / 2) - 1
phy_timing = phy_timing[:, 0:2]
timing = {}
timing["phy"] = phy_timing
timing["gen"] = gen_timing


# run preprocessing once per run per subject
for subject in subjects:
    try:
        print subject
        ntwk_run_cond = {}
        ntwk = {}
        hipp = {}
        hipp_run_cond = {}
        corrmats = {}
        for run in runs:
            # xfm laird 2011 maps to subject's epi space & define masker
            epi = join(
                data_dir, subject, "{0}-{1}_retr-mcf.nii.gz".format(subject, run)
            )
            confounds = join(
                data_dir, subject, "{0}-{1}_retr-confounds.txt".format(subject, run)
            )
            # icn = join(data_dir, subject,'{0}-{1}_18_icn_retr.nii.gz'.format(subject, run))
            # icn_regions = connected_label_regions(icn, min_size=50., labels=labels)
            icn_regions = join(
                data_dir,
                subject,
                "{0}-{1}_18_icn-regions_retr.nii.gz".format(subject, run),
            )
            hippo = join(
                data_dir, subject, "{0}-{1}_hippo_retr.nii.gz".format(subject, run)
            )
            regn_masker = NiftiLabelsMasker(
                icn_regions, standardize=True, high_pass=highpass, t_r=2.0, verbose=1
            )
            hipp_masker = NiftiLabelsMasker(
                hippo, standardize=True, high_pass=highpass, t_r=2.0, verbose=1
            )

            # extract the network-wise and hippocampus timeseries per run
            # fmri = join(data_dir, subject, 'session-1', 'retr', 'mni', '{0}_filtered_func_data_{1}.nii.gz'.format(subject, run))
            ntwk_ts = regn_masker.fit_transform(epi, confounds=confounds)
            hipp_ts = hipp_masker.fit_transform(epi, confounds=confounds)
            # ts = [ntwk_ts, hipp_ts]
            # and then separate each run's timeseries into the different conditions
            for condition in conditions:
                ntwk_run_cond["{0} {1}".format(condition, run)] = np.vstack(
                    (
                        ntwk_ts[
                            timing[condition][0, 0]
                            .astype(int) : (
                                timing[condition][0, 0] + timing[condition][0, 1] + 1
                            )
                            .astype(int),
                            :,
                        ],
                        ntwk_ts[
                            timing[condition][1, 0]
                            .astype(int) : (
                                timing[condition][1, 0] + timing[condition][1, 1] + 1
                            )
                            .astype(int),
                            :,
                        ],
                        ntwk_ts[
                            timing[condition][2, 0]
                            .astype(int) : (
                                timing[condition][2, 0] + timing[condition][2, 1] + 1
                            )
                            .astype(int),
                            :,
                        ],
                    )
                )
                print ntwk_run_cond["{0} {1}".format(condition, run)].shape
                hipp_run_cond["{0} {1}".format(condition, run)] = np.vstack(
                    (
                        hipp_ts[
                            timing[condition][0, 0]
                            .astype(int) : (
                                timing[condition][0, 0] + timing[condition][0, 1] + 1
                            )
                            .astype(int)
                        ],
                        hipp_ts[
                            timing[condition][1, 0]
                            .astype(int) : (
                                timing[condition][1, 0] + timing[condition][1, 1] + 1
                            )
                            .astype(int)
                        ],
                        hipp_ts[
                            timing[condition][2, 0]
                            .astype(int) : (
                                timing[condition][2, 0] + timing[condition][2, 1] + 1
                            )
                            .astype(int)
                        ],
                    )
                )
        for condition in conditions:

            ntwk[condition] = np.vstack(
                (
                    ntwk_run_cond["{0} 0".format(condition)],
                    ntwk_run_cond["{0} 1".format(condition)],
                )
            )
            hipp[condition] = np.vstack(
                (
                    hipp_run_cond["{0} 0".format(condition)],
                    hipp_run_cond["{0} 1".format(condition)],
                )
            )
            corrmats[condition] = correlation_measure.fit_transform([ntwk[condition]])[
                0
            ]
            df = pd.DataFrame(corrmats[condition], index=labels, columns=labels)
            df.to_csv(
                join(
                    sink_dir,
                    "{0}-{1}-corrmat-regionwise.csv".format(subject, condition),
                )
            )
            df.to_csv(
                join(
                    data_dir,
                    subject,
                    "{0}-{1}-corrmat-regionwise.csv".format(subject, condition),
                )
            )
    except Exception as e:
        print e
