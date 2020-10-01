from __future__ import division
from os.path import join, basename, exists

import numpy as np
import pandas as pd
from datetime import datetime


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
# subjects = ['101','103']


# data_dir = '/home/data/nbc/physics-learning/data/pre-processed'
data_dir = "/home/data/nbc/physics-learning/retrieval-graphtheory/output"
preroc_dir = "/home/data/nbc/physics-learning/data/pre-processed"
timing_dir = "/home/data/nbc/physics-learning/data/behavioral-data/vectors"

sessions = ["pre", "post"]

tasks = {
    "reas": [{"conditions": ["Reasoning", "Baseline"]}, {"runs": [0, 1]}],
    "retr": [{"conditions": ["Physics", "General"]}, {"runs": [0, 1]}],
    "fci": [{"conditions": ["Physics", "NonPhysics"]}, {"runs": [0, 1, 2]}],
}

conds = ["high-level", "lower-level"]

index = pd.MultiIndex.from_product(
    [subjects, sessions, tasks.keys(), conds],
    names=["subject", "session", "task", "condition"],
)
df = pd.DataFrame(columns=["average fd"], index=index, dtype=np.float64)

for subject in subjects:
    for i in np.arange(0, len(sessions)):
        for task in tasks.keys():
            conditions = tasks[task][0]["conditions"]
            fd_cond = {}
            timing = {}
            sliced_fd = {}
            try:
                for run in tasks[task][1]["runs"]:
                    for condition in conditions:
                        print (task, run, condition)
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
                                        "{0}-{1}-{2}.txt".format(task, run, condition),
                                    ),
                                    delimiter="\t",
                                    dtype="float",
                                )
                            timing["{0}-{1}".format(run, condition)][:, 0] = (
                                np.round(
                                    timing["{0}-{1}".format(run, condition)][:, 0] / 2,
                                    0,
                                )
                                - 1
                            )
                            timing["{0}-{1}".format(run, condition)][:, 1] = np.round(
                                np.round(
                                    timing["{0}-{1}".format(run, condition)][:, 1], 0
                                )
                                / 2,
                                0,
                            )
                            timing["{0}-{1}".format(run, condition)] = timing[
                                "{0}-{1}".format(run, condition)
                            ][:, 0:2]
                            # print(timing['{0}-{1}'.format(run, condition)])
                        else:
                            # make this work better for reasoning timing
                            timing["{0}-{1}".format(run, condition)] = np.genfromtxt(
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
                                    timing["{0}-{1}".format(run, condition)][:, 0] / 2,
                                    0,
                                )
                                - 2
                            )
                            timing["{0}-{1}".format(run, condition)][:, 1] = 3
                            timing["{0}-{1}".format(run, condition)] = timing[
                                "{0}-{1}".format(run, condition)
                            ][:, 0:2]
                            # print(timing['{0}-{1}'.format(run, condition)])
                    if task == "fci":
                        fd = np.genfromtxt(
                            join(
                                preroc_dir,
                                subject,
                                "session-{0}".format(i),
                                "fci/fci-{0}/fci-{0}-ppi.feat".format(run),
                                "{0}-session-{1}-fd.txt".format(subject, i),
                            )
                        )
                    else:
                        fd = np.genfromtxt(
                            join(
                                data_dir,
                                sessions[i],
                                subject,
                                "{0}-session-{1}_{2}-{3}_fd.txt".format(
                                    subject, i, task, run
                                ),
                            )
                        )
                    for condition in conditions:
                        print (
                            fd[
                                timing["{0}-{1}".format(run, condition)][0, 0]
                                .astype(int) : (
                                    timing["{0}-{1}".format(run, condition)][0, 0]
                                    + timing["{0}-{1}".format(run, condition)][0, 1]
                                    + 1
                                )
                                .astype(int)
                            ].shape
                        )
                        print (
                            fd[
                                timing["{0}-{1}".format(run, condition)][1, 0]
                                .astype(int) : (
                                    timing["{0}-{1}".format(run, condition)][1, 0]
                                    + timing["{0}-{1}".format(run, condition)][1, 1]
                                    + 1
                                )
                                .astype(int)
                            ].shape
                        )
                        print (
                            fd[
                                timing["{0}-{1}".format(run, condition)][2, 0]
                                .astype(int) : (
                                    timing["{0}-{1}".format(run, condition)][2, 0]
                                    + timing["{0}-{1}".format(run, condition)][2, 1]
                                    + 1
                                )
                                .astype(int)
                            ].shape
                        )
                        fd_cond["{0}-{1}".format(run, condition)] = np.hstack(
                            (
                                fd[
                                    timing["{0}-{1}".format(run, condition)][0, 0]
                                    .astype(int) : (
                                        timing["{0}-{1}".format(run, condition)][0, 0]
                                        + timing["{0}-{1}".format(run, condition)][0, 1]
                                        + 1
                                    )
                                    .astype(int)
                                ],
                                fd[
                                    timing["{0}-{1}".format(run, condition)][1, 0]
                                    .astype(int) : (
                                        timing["{0}-{1}".format(run, condition)][1, 0]
                                        + timing["{0}-{1}".format(run, condition)][1, 1]
                                        + 1
                                    )
                                    .astype(int)
                                ],
                                fd[
                                    timing["{0}-{1}".format(run, condition)][2, 0]
                                    .astype(int) : (
                                        timing["{0}-{1}".format(run, condition)][2, 0]
                                        + timing["{0}-{1}".format(run, condition)][2, 1]
                                        + 1
                                    )
                                    .astype(int)
                                ],
                            )
                        )
                        print (
                            "cut fd for {0}, run {1}, {2}".format(task, run, condition),
                            fd_cond["{0}-{1}".format(run, condition)].shape,
                        )
                # and paste together the fd from each run together per condition
                for j in np.arange(0, len(conditions)):
                    print task, conditions[j], "pasting fd together per condition"
                    if task != "fci":
                        sliced_fd[conditions[j]] = np.hstack(
                            (
                                fd_cond["0-{0}".format(conditions[j])],
                                fd_cond["1-{0}".format(conditions[j])],
                            )
                        )
                        print (sliced_fd[conditions[j]].shape)
                    else:
                        sliced_fd[conditions[j]] = np.hstack(
                            (
                                fd_cond["0-{0}".format(conditions[j])],
                                fd_cond["1-{0}".format(conditions[j])],
                                fd_cond["2-{0}".format(conditions[j])],
                            )
                        )
                        print (sliced_fd[conditions[j]].shape)
                for j in np.arange(0, len(conditions)):
                    df.loc[
                        (subject, sessions[i], task, conds[j]), "average fd"
                    ] = np.average(sliced_fd[conditions[j]])
            except Exception as e:
                print (subject, task, e)
df.to_csv(
    join(data_dir, "avg-fd-per-condition-per-run_{0}.csv".format(str(datetime.now())))
)
