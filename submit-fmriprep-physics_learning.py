from bids.layout import BIDSLayout
from glob import glob
import os.path as op
import subprocess
import random
import string

layout = BIDSLayout('/scratch/kbott/physics-learning_dset/')
subjects = layout.get_subjects()

with open('submit-fmriprep-physics_learning.sub', 'r') as fo:
    data = fo.read()

for sub in subjects:
    work = 'work/{0}-fmriprep-work'.format(sub)
    sub_data = data.format(sub=sub, work=work)

    file_ = 'jobfiles/physics-learning_fmriprep_sub-{0}.sub'.format(sub)
    with open(file_, 'w') as fo:
        fo.write(sub_data)

    sub_str = 'bsub<{0}'.format(file_)
    process = subprocess.Popen(sub_str, shell=True)
