#!/usr/bin/env python3

"""
Phys2bids is a python3 library meant to set physiological files in BIDS standard.
It was born for Acqknowledge files (BIOPAC), and at the moment it supports
``.acq`` files and ``.txt`` files obtained by labchart (ADInstruments).
It requires python 3.6 or above, as well as the modules:
- `numpy`
- `matplotlib`
In order to process ``.acq`` files, it needs `bioread`, an excellent module
that can be found at `this link`_
The project is under development.
At the very moment, it assumes:
-  the input file is from one individual scan, not one session with multiple scans.
.. _this link:
   https://github.com/uwmadison-chm/bioread
Copyright 2019, The Phys2BIDS community.
Please scroll to bottom to read full license.
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import bids
import argparse
import logging
from os import makedirs
from os.path import join, exists, basename
from glob import glob
from nilearn import input_data, connectome, plotting, image

from idconn.connectivity.build_networks import connectivity, task_connectivity
#from idconn.networking import graph_theory, null_distribution

LGR = logging.getLogger(__name__)
LGR.setLevel(logging.INFO)
print('Getting started!')

parser = argparse.ArgumentParser(description='Make correlation matrices from BOLD data + mask.')
parser.add_argument('dset_dir', type=str, 
                    help='Path to BIDS dataset containing fmriprep derivatives folder.')
parser.add_argument('atlas', type=str, 
                    help='Path to atlas file in space specified by `space`.')
parser.add_argument('task', type=str, 
                    help='Task to be analyzed.')
parser.add_argument('--out_dir', type=str, help='Overwrites automatic idconn derivatives path.')

parser.add_argument('--space', type=str,
                    help='Space in which to run analyses (must be the space `atlas` is in.')
parser.add_argument('--conn', type=str, 
                    help='Metric used to calculate connectivity. Must be one of {“covariance”, “correlation”, “partial correlation”, “tangent”, “precision”}.')
parser.add_argument('--bids_db', type=str, help='Path to saved BIDS dataset layout file.')
parser.add_argument('--confounds', nargs="+", help='Names of confound regressors from ')

args = parser.parse_args()

if not args.confounds:
    confounds = ["cosine00", "cosine01", "cosine02", "trans_x", "trans_x_derivative1", "trans_x_power2", "trans_x_derivative1_power2", "trans_y", "trans_y_derivative1", "trans_y_derivative1_power2", "trans_y_power2", "trans_z", "trans_z_derivative1", "trans_z_power2", "trans_z_derivative1_power2", "rot_x", "rot_x_derivative1", "rot_x_power2", "rot_x_derivative1_power2", "rot_y", "rot_y_derivative1", "rot_y_power2", "rot_y_derivative1_power2", "rot_z", "rot_z_derivative1", "rot_z_derivative1_power2", "rot_z_power2", "a_comp_cor_00", "a_comp_cor_01", "a_comp_cor_02", "a_comp_cor_03", "a_comp_cor_04", "a_comp_cor_05", "a_comp_cor_06"]
else:
    confounds = args.confounds

if not args.space:
    space = 'MNI152NLin2009cAsym'
else:
    space = args.space

if not args.conn:
    conn = 'correlation'
else:
    conn = args.conn
print(f"Atlas: {args.atlas}\nConnectivity measure: {conn}")

if args.bids_db:
    bids_db = args.bids_db
else:
    bids_db = None

assert exists(args.dset_dir), "Specified dataset doesn't exist:\n{dset_dir} not found.\n\nPlease check the filepath."
layout = bids.BIDSLayout(args.dset_dir, derivatives=True, database_path=bids_db)
subjects = layout.get(return_type='id', target='subject', suffix='bold')
print(f"Subjects: {subjects}")
#runs = layout.get(return_type='id', target='session', suffix='bold')
preproc_subjects = layout.get(return_type='id', target='subject', task=args.task, space=space, desc='preproc', suffix='bold')
if len(subjects) != len(preproc_subjects):
    print(f'{len(subjects)} subjects found in dset, only {len(preproc_subjects)} have preprocessed BOLD data. Pipeline is contniuing anyway, please double check preprocessed data if this doesn\'t seem right.')

example_events = layout.get(return_type='filename', suffix='events', task=task, subject=preproc_subjects[0])
events_df = pd.read_csv(example_events[0])
conditions = events_df['trial_type'].unique()

print(f"Computing connectivity matrices using {args.atlas}")
for subject in preproc_subjects:
    print(f"Subject {subject}")
    sessions = layout.get(return_type='id', target='session', task=args.task, subject=subject, suffix='bold')
    print(f"Sessions with task-{args.task} found for {subject}: {sessions}")
    for session in sessions:
        print(f"Session {session}")
        if 'rest' in task:
            adj_matrix = connectivity(layout, subject, session, args.task, args.atlas, conn, space, confounds)
        if len(conditions) < 1:
            adj_matrix = connectivity(layout, subject, session, args.task, args.atlas, conn, space, confounds)
        else:
            adj_matrix = task_connectivity(layout, subject, session, args.task, args.atlas, conn, space, confounds)

def _main(argv=None):
    options = _get_parser().parse_args(argv)
    idconn(**vars(options))


if __name__ == "__main__":
    _main(sys.argv[1:])

"""
Copyright 2019, The Phys2BIDS community.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
