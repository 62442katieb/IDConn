import numpy as np
import pandas as pd
from os.path import join
from datetime import datetime
from nilearn.input_data import NiftiMasker

data_dir = "/home/data/nbc/physics-learning/data/pre-processed"
sink_dir = "/home/data/nbc/physics-learning/retrieval-graphtheory/output"

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
# subjects = ['101', '103']

sessions = ["pre", "post"]

df = pd.DataFrame(index=subjects, columns=["average_head_size"])

masker = NiftiMasker()

for subject in subjects:
    head_size = 0
    for i in np.arange(0, len(sessions)):
        bet_mask = join(
            data_dir,
            subject,
            "session-{0}".format(i),
            "anatomical/anatomical-0/fsl/anatomical-bet_mask.nii.gz",
        )
        mask_array = masker.fit_transform(bet_mask)
        head_size += np.sum(mask_array)
    df.at[subject] = head_size / 2
df.to_csv(join(sink_dir, "head-size_{0}.csv".format(str(datetime.now()))))
