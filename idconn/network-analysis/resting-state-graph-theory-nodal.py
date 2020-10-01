import numpy as np
import pandas as pd
from os import makedirs
from os.path import join, exists

# from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.plotting import plot_anat, plot_roi
import bct

# from nipype.interfaces.fsl import InvWarp, ApplyWarp
import datetime

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


sink_dir = "/Users/katherine/Dropbox/Projects/physics-retrieval/data/output"

shen = "/home/kbott006/physics-retrieval/shen2015_2mm_268_parcellation.nii.gz"
craddock = "/home/kbott006/physics-retrieval/craddock2012_tcorr05_2level_270_2mm.nii.gz"
masks = {"shen2015": shen, "craddock2012": craddock}

sessions = [0, 1]
sesh = ["pre", "post"]
tasks = ["rest"]

kappa_upper = 0.21
kappa_lower = 0.31

lab_notebook_dir = sink_dir
index = pd.MultiIndex.from_product([subjects, sessions], names=["subject", "session"])
lab_notebook = pd.DataFrame(index=index, columns=["start", "end", "errors"])

correlation_measure = ConnectivityMeasure(kind="correlation")


index = pd.MultiIndex.from_product(
    [subjects, sessions, tasks, masks.keys()],
    names=["subject", "session", "task", "mask"],
)
df = pd.DataFrame(columns=["lEff0", "clustCoeff0"], index=index, dtype=np.float64)

for subject in subjects:
    for session in sessions:
        lab_notebook.at[(subject, session), "start"] = str(datetime.datetime.now())
        for task in tasks:
            for mask in masks.keys():
                try:
                    # shen_masker = NiftiLabelsMasker(xfmd_masks['shen2015'], background_label=0, standardize=True, detrend=True,t_r=3.)
                    # craddock_masker = NiftiLabelsMasker(xfmd_masks['craddock2012'], background_label=0, standardize=True, detrend=True,t_r=3.)

                    # confounds = '/home/data/nbc/physics-learning/anxiety-physics/output/{1}/{0}/{0}_confounds.txt'.format(subject, sesh[session])
                    # epi_data = join(data_dir, subject, 'session-{0}'.format(session), 'resting-state/resting-state-0/endor1.feat', 'filtered_func_data.nii.gz')

                    # shen_ts = shen_masker.fit_transform(epi_data, confounds)
                    # shen_corrmat = correlation_measure.fit_transform([shen_ts])[0]
                    # np.savetxt(join(sink_dir, sesh[session], subject, '{0}-session-{1}-rest_network_corrmat_shen2015.csv'.format(subject, session)), shen_corrmat, delimiter=",")
                    corrmat = np.genfromtxt(
                        join(
                            sink_dir,
                            "{0}-session-{1}-{2}_network_corrmat_{3}.csv".format(
                                subject, session, task, mask
                            ),
                        ),
                        delimiter=",",
                    )
                    print(corrmat.shape)
                    # craddock_ts = craddock_masker.fit_transform(epi_data, confounds)
                    # craddock_corrmat = correlation_measure.fit_transform([craddock_ts])[0]
                    # np.savetxt(join(sink_dir, sesh[session], subject, '{0}-session-{1}-rest_network_corrmat_craddock2012.csv'.format(subject, session)), craddock_corrmat, delimiter=",")

                    ge_s = []
                    ge_c = []

                    md_s = []
                    md_c = []
                    for p in np.arange(kappa_upper, kappa_lower, 0.01):
                        thresh = bct.threshold_proportional(corrmat, p, copy=True)

                        # network measures of interest here
                        # global efficiency
                        ge = bct.efficiency_wei(thresh, local=True)
                        ge_s.append(ge)

                        # modularity
                        md = bct.clustering_coef_wu(thresh)
                        md_s.append(md)

                    ge_s = np.asarray(ge_s)
                    md_s = np.asarray(md_s)
                    leff = np.trapz(ge_s, dx=0.01, axis=0)
                    print("local efficiency:", leff[0])
                    ccoef = np.trapz(md_s, dx=0.01, axis=0)
                    for j in np.arange(1, 270):
                        df.at[
                            (subject, session, task, mask), "lEff{0}".format(j)
                        ] = leff[j - 1]
                        df.at[
                            (subject, session, task, mask), "clustCoeff{0}".format(j)
                        ] = ccoef[j - 1]
                    # df.to_csv(join(sink_dir, 'resting-state_graphtheory_shen+craddock.csv'), sep=',')
                    lab_notebook.at[(subject, session), "end"] = str(
                        datetime.datetime.now()
                    )
                except Exception as e:
                    print(e, subject, session)
                    lab_notebook.at[(subject, session), "errors"] = [
                        e,
                        str(datetime.datetime.now()),
                    ]
        df.to_csv(
            join(sink_dir, "resting-state_graphtheory_shen+craddock.csv"), sep=","
        )

df.to_csv(
    join(
        sink_dir,
        "resting-state_graphtheory_shen+craddock_{0}.csv".format(
            str(datetime.datetime.today())
        ),
    ),
    sep=",",
)
lab_notebook.to_csv(
    join(
        lab_notebook_dir,
        "LOG_resting-state-graph-theory_{0}.csv".format(str(datetime.datetime.now())),
    )
)
