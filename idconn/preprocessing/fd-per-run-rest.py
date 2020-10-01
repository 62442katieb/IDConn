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
preproc_dir = "/home/data/nbc/physics-learning/data/pre-processed"

sessions = ["pre", "post"]


index = pd.MultiIndex.from_product([subjects, sessions], names=["subject", "session"])
df = pd.DataFrame(columns=["average fd"], index=index, dtype=np.float64)

for subject in subjects:
    for i in np.arange(0, len(sessions)):
        try:
            df.loc[(subject, sessions[i]), "average fd"] = np.genfromtxt(
                join(
                    preproc_dir,
                    subject,
                    "session-{0}/resting-state/resting-state-0/endor1.feat/mc/prefiltered_func_data_mcf_abs_mean.rms".format(
                        i
                    ),
                )
            )
        except Exception as e:
            print(subject, sessions[i], e)
df.to_csv(join(data_dir, "avg-fd-per-run-rest_{0}.csv".format(str(datetime.now()))))
