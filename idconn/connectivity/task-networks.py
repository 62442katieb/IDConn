from __future__ import division
from os.path import join, basename, exists
from os import makedirs
from glob import glob

from nilearn import input_data, datasets, plotting, regions
from nilearn.image import concat_imgs
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from scipy.stats import pearsonr, skew

from nipype.interfaces.fsl import ApplyWarp, InvWarp

import bct
import json
import numpy as np
import pandas as pd
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
    "214",
    "215",
    "216",
    "217",
    "218",
    "219",
    "320",
    "321",
    "323",
    "324",
    "325",
    "327",
    "328",
    "330",
    "331",
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
    "453",
    "455",
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
    "577",
    "578",
    "581",
    "582",
    "584",
    "585",
    "586",
    "587",
    "588",
    "589",
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
    "612",
    "613",
    "614",
    "615",
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
    "629",
    "630",
    "631",
    "633",
    "634",
]
# all subjects 102 103 101 104 106 107 108 110 212 X213 214 215 216 217 218 219 320 321 X322 323 324 325
# 327 328 X329 330 331 X332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 451
# X452 453 455 X456 X457 458 459 460 462 463 464 465 467 468 469 470 502 503 571 572 573 574 X575 577 578
# X579 X580 581 582 584 585 586 587 588 589 X590 591 592 593 594 595 596 597 598 604 605 606 607 608 609
# 610 X611 612 613 614 615 X616 617 618 619 620 621 622 623 624 625 626 627 X628 629 630 631 633 634
# errors in fnirt-to-mni: 213, 322, 329, 332, 452, 456, 457, 575, 579, 580, 590, 611, 616, 628
# subjects without post-IQ measure: 452, 461, 501, 575, 576, 579, 583, 611, 616, 628, 105, 109, 211, 213, 322, 326, 329, 332
subjects = ["101", "103"]


# data_dir = '/home/data/nbc/physics-learning/data/pre-processed'
data_dir = "/home/data/nbc/physics-learning/retrieval-graphtheory/output"
exfunc_dir = "/home/data/nbc/physics-learning/data/pre-processed"
timing_dir = "/home/data/nbc/physics-learning/data/behavioral-data/vectors"

sessions = ["pre", "post"]

tasks = {
    "reas": [{"conditions": ["Reasoning", "Baseline"]}, {"runs": [0, 1]}],
    "retr": [{"conditions": ["Physics", "General"]}, {"runs": [0, 1]}],
    "fci": [{"conditions": ["Physics", "NonPhysics"]}, {"runs": [0, 1, 2]}],
}

masks = {
    "shen2015": "/home/kbott006/physics-retrieval/shen2015_2mm_268_parcellation.nii.gz",
    "craddock2012": "/home/kbott006/physics-retrieval/craddock2012_tcorr05_2level_270_2mm.nii.gz",
}

connectivity_metric = "correlation"
conds = ["high-level", "lower-level"]

# find a way to estimate this threshold range...
# or threshold it
thresh_range = np.arange(0.1, 1, 0.1)

# this should be calculated from the task timing ughhhhhhh
highpass = 1 / 55.0

correlation_measure = ConnectivityMeasure(kind=connectivity_metric)


index = pd.MultiIndex.from_product(
    [subjects, sessions, tasks.keys(), conds, masks],
    names=["subject", "session", "task", "condition", "mask"],
)
df = pd.DataFrame(
    columns=["k_scale-free", "k_connected"], index=index, dtype=np.float64
)

for subject in subjects:
    print (subject)
    try:
        for i in np.arange(0, len(sessions)):
            print i
            run_cond = {}
            for task in tasks.keys():
                print task
                timing = {}
                conditions = tasks[task][0]["conditions"]
                for mask in masks.keys():
                    print mask
                    sliced_ts = {}
                    for run in tasks[task][1]["runs"]:
                        print run
                        mask_file = join(
                            data_dir,
                            sessions[i],
                            subject,
                            "{0}-session-{1}_{2}-{3}_{4}.nii.gz".format(
                                subject, i, task, run, mask
                            ),
                        )
                        print (mask_file)
                        if task == "fci":
                            if not exists(mask_file):
                                print (
                                    mask_file,
                                    "doesn't exist, so we're gonna make one",
                                )
                                try:
                                    mni2epiwarp = join(
                                        data_dir,
                                        sessions[i],
                                        subject,
                                        "{0}-session-{1}_{2}-{3}_mni-fnirt-epi-warp.nii.gz".format(
                                            subject, i, task, run
                                        ),
                                    )
                                    example_func_file = "/home/data/nbc/physics-learning/data/pre-processed/{0}/session-{1}/fci/fci-{2}/fci-{2}-ppi.feat/reg/example_func.nii.gz".format(
                                        subject, i, run
                                    )
                                    example_func2standard = "/home/data/nbc/physics-learning/data/pre-processed/{0}/session-{1}/fci/fci-{2}/fci-{2}-ppi.feat/reg/example_func2standard_warp.nii.gz".format(
                                        subject, i, run
                                    )
                                    print example_func2standard
                                    print mask
                                    print masks[mask]
                                    warpspeed = ApplyWarp(
                                        interp="nn", output_type="NIFTI_GZ"
                                    )
                                    if not exists(mni2epiwarp):
                                        # invert the epi-to-mni warpfield so you can run these analyses in native space
                                        invert = InvWarp(output_type="NIFTI_GZ")
                                        invert.inputs.warp = example_func2standard
                                        invert.inputs.inverse_warp = mni2epiwarp
                                        invert.inputs.reference = example_func_file
                                        inverted = invert.run()
                                        warpspeed.inputs.field_file = (
                                            inverted.outputs.inverse_warp
                                        )
                                    else:
                                        warpspeed.inputs.ref_file = example_func_file
                                        warpspeed.inputs.field_file = mni2epiwarp
                                        warpspeed.inputs.in_file = masks[mask]
                                        warpspeed.inputs.out_file = mask_file
                                        warped = warpspeed.run()

                                    display = plotting.plot_roi(
                                        mask_file,
                                        bg_img=example_func_file,
                                        colorbar=True,
                                    )
                                    display.savefig(
                                        join(
                                            data_dir,
                                            "qa",
                                            "{0}-session-{1}_fci-{2}_qa_{3}.png".format(
                                                subject, i, run, mask
                                            ),
                                        ),
                                        dpi=300,
                                    )
                                    display.close()
                                except Exception as e:
                                    print ("mni2epiwarp not finished for", mask, ":", e)
                        else:
                            pass
                        for condition in conditions:
                            print subject, task, run, condition
                            if task != "reas":
                                if task == "retr":
                                    timing[
                                        "{0}-{1}".format(run, condition)
                                    ] = np.genfromtxt(
                                        join(
                                            "/home/data/nbc/physics-learning/retrieval-graphtheory/",
                                            "RETRcondition{0}Sess{1}.txt".format(
                                                condition, i
                                            ),
                                        ),
                                        delimiter="\t",
                                        dtype="float",
                                    )
                                if task == "fci":
                                    timing[
                                        "{0}-{1}".format(run, condition)
                                    ] = np.genfromtxt(
                                        join(
                                            timing_dir,
                                            subject,
                                            "session-{0}".format(i),
                                            task,
                                            "{0}-{1}-{2}.txt".format(
                                                task, run, condition
                                            ),
                                        ),
                                        delimiter="\t",
                                        dtype="float",
                                    )
                                timing["{0}-{1}".format(run, condition)][:, 0] = (
                                    np.round(
                                        timing["{0}-{1}".format(run, condition)][:, 0]
                                        / 2,
                                        0,
                                    )
                                    - 1
                                )
                                timing["{0}-{1}".format(run, condition)][
                                    :, 1
                                ] = np.round(
                                    np.round(
                                        timing["{0}-{1}".format(run, condition)][:, 1],
                                        0,
                                    )
                                    / 2,
                                    0,
                                )
                                timing["{0}-{1}".format(run, condition)] = timing[
                                    "{0}-{1}".format(run, condition)
                                ][:, 0:2]
                                highpass = np.average(
                                    timing["{0}-{1}".format(run, condition)][:, 1]
                                ) * len(conditions)
                                print (timing["{0}-{1}".format(run, condition)])
                            else:
                                highpass = 1 / 66.0
                                # make this work better for reasoning timing
                                timing[
                                    "{0}-{1}".format(run, condition)
                                ] = np.genfromtxt(
                                    join(
                                        timing_dir,
                                        subject,
                                        "session-{0}".format(i),
                                        task,
                                        "{0}-{1}-{2}.txt".format(task, run, condition),
                                    ),
                                    delimiter="\t",
                                    dtype="float",
                                )
                                # print(np.average(timing['{0}-{1}'.format(run, condition)][:,1]))
                                timing["{0}-{1}".format(run, condition)][:, 0] = (
                                    np.round(
                                        timing["{0}-{1}".format(run, condition)][:, 0]
                                        / 2,
                                        0,
                                    )
                                    - 2
                                )
                                timing["{0}-{1}".format(run, condition)][:, 1] = 3
                                timing["{0}-{1}".format(run, condition)] = timing[
                                    "{0}-{1}".format(run, condition)
                                ][:, 0:2]
                                # print(timing['{0}-{1}'.format(run, condition)])
                        print (highpass)
                        epi = join(
                            data_dir,
                            sessions[i],
                            subject,
                            "{0}-session-{1}_{2}-{3}_mcf.nii.gz".format(
                                subject, i, task, run
                            ),
                        )
                        confounds = join(
                            data_dir,
                            sessions[i],
                            subject,
                            "{0}-session-{1}_{2}-{3}_mcf.nii.gz.par".format(
                                subject, i, task, run
                            ),
                        )
                        assert exists(epi), "epi_mcf does not exist at {0}".format(epi)
                        print epi
                        assert exists(
                            confounds
                        ), "confounds+outliers.txt does not exist at {0}".format(
                            confounds
                        )
                        print confounds

                        # for each parcellation, extract BOLD timeseries
                        masker = NiftiLabelsMasker(
                            mask_file,
                            standardize=True,
                            high_pass=highpass,
                            t_r=2.0,
                            verbose=1,
                        )
                        timeseries = masker.fit_transform(epi, confounds)

                        # and now we slice into conditions
                        for condition in conditions:
                            if not exists(
                                join(
                                    data_dir,
                                    sessions[i],
                                    subject,
                                    "{0}-session-{1}_{2}-{3}_{4}-corrmat.csv".format(
                                        subject, i, task, condition, mask
                                    ),
                                )
                            ):
                                print (
                                    "{0}-{1}-{2}-{3}".format(task, run, condition, mask)
                                )
                                run_cond[
                                    "{0}-{1}-{2}".format(task, run, condition)
                                ] = np.vstack(
                                    (
                                        timeseries[
                                            timing["{0}-{1}".format(run, condition)][
                                                0, 0
                                            ]
                                            .astype(int) : (
                                                timing[
                                                    "{0}-{1}".format(run, condition)
                                                ][0, 0]
                                                + timing[
                                                    "{0}-{1}".format(run, condition)
                                                ][0, 1]
                                                + 1
                                            )
                                            .astype(int),
                                            :,
                                        ],
                                        timeseries[
                                            timing["{0}-{1}".format(run, condition)][
                                                1, 0
                                            ]
                                            .astype(int) : (
                                                timing[
                                                    "{0}-{1}".format(run, condition)
                                                ][1, 0]
                                                + timing[
                                                    "{0}-{1}".format(run, condition)
                                                ][1, 1]
                                                + 1
                                            )
                                            .astype(int),
                                            :,
                                        ],
                                        timeseries[
                                            timing["{0}-{1}".format(run, condition)][
                                                2, 0
                                            ]
                                            .astype(int) : (
                                                timing[
                                                    "{0}-{1}".format(run, condition)
                                                ][2, 0]
                                                + timing[
                                                    "{0}-{1}".format(run, condition)
                                                ][2, 1]
                                                + 1
                                            )
                                            .astype(int),
                                            :,
                                        ],
                                    )
                                )
                                print (
                                    "extracted signals for {0}, run {1}, {2}".format(
                                        task, run, condition
                                    ),
                                    run_cond[
                                        "{0}-{1}-{2}".format(task, run, condition)
                                    ].shape,
                                )
                            else:
                                pass
                    # and paste together the timeseries from each run together per condition
                    for j in np.arange(0, len(conditions)):
                        if not exists(
                            join(
                                data_dir,
                                sessions[i],
                                subject,
                                "{0}-session-{1}_{2}-{3}_{4}-corrmat.csv".format(
                                    subject, i, task, conditions[j], mask
                                ),
                            )
                        ):
                            print task, conditions[
                                j
                            ], "pasting timeseries together per condition"
                            if task != "fci":
                                # print('task isn\'t FCI, only 2 runs')
                                # print('{0}-0-{1}'.format(task, conditions[j]))
                                sliced_ts[conditions[j]] = np.vstack(
                                    (
                                        run_cond[
                                            "{0}-0-{1}".format(task, conditions[j])
                                        ],
                                        run_cond[
                                            "{0}-1-{1}".format(task, conditions[j])
                                        ],
                                    )
                                )
                                print (sliced_ts[conditions[j]].shape)
                            else:
                                # print('task is FCI, 3 runs')
                                sliced_ts[conditions[j]] = np.vstack(
                                    (
                                        run_cond["fci-0-{0}".format(conditions[j])],
                                        run_cond["fci-1-{0}".format(conditions[j])],
                                        run_cond["fci-2-{0}".format(conditions[j])],
                                    )
                                )
                                print (sliced_ts[conditions[j]].shape)
                            corrmat = correlation_measure.fit_transform(
                                [sliced_ts[conditions[j]]]
                            )[0]
                            print (corrmat.shape)
                            np.savetxt(
                                join(
                                    data_dir,
                                    sessions[i],
                                    subject,
                                    "{0}-session-{1}_{2}-{3}_{4}-corrmat.csv".format(
                                        subject, i, task, conditions[j], mask
                                    ),
                                ),
                                corrmat,
                            )

                            # reset kappa starting point
                            # calculate proportion of connections that can be retained
                            # before degree dist. ceases to be scale-free
                            kappa = 0.01
                            skewness = 1
                            while abs(skewness) > 0.3:
                                w = bct.threshold_proportional(corrmat, kappa)
                                skewness = skew(bct.degrees_und(w))
                                kappa += 0.01
                            df.at[
                                (subject, sessions[i], task, conds[j], mask),
                                "k_scale-free",
                            ] = kappa

                            # reset kappa starting point
                            # calculate proportion of connections that need to be retained
                            # for node connectedness
                            kappa = 0.01
                            num = 2
                            while num > 1:
                                w = bct.threshold_proportional(corrmat, kappa)
                                [comp, num] = bct.get_components(w)
                                num = np.unique(comp).shape[0]
                                kappa += 0.01
                            df.at[
                                (subject, sessions[i], task, conds[j], mask),
                                "k_connected",
                            ] = kappa
                        else:
                            pass
                            # df.at[(subject, sessions[i], task, conds[j], mask),'k_connected'] = kappa
    except Exception as e:
        print (subject, "didn't run, because", e)

df.to_csv(
    join(
        data_dir,
        "phys-learn-fci_kappa_{0}-{1}.csv".format(
            connectivity_metric, str(datetime.datetime.now())
        ),
    ),
    sep=",",
)
