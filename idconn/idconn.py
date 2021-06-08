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

import datetime
import logging
import os
import sys
import argparse
from copy import deepcopy
from shutil import copy as cp

import numpy as np

from idconn import io, _version

# from idconn.cli.run import _get_parser
from idconn.connectivity import build_networks

# from idconn.networking import graph_measures
# from idconn.data import
# from idconn.statistics import

from . import __version__
from .due import due, Doi

LGR = logging.getLogger(__name__)
LGR.setLevel(logging.INFO)


def _get_parser():
    """
    Parse command line inputs for this function.
    Returns
    -------
    parser.parse_args() : argparse dict
    Notes
    -----
    # Argument parser follow template provided by RalphyZ.
    # https://stackoverflow.com/a/43456577
    """
    parser = argparse.ArgumentParser()
    optional = parser._action_groups.pop()
    required = parser.add_argument_group("Required Argument:")
    required.add_argument(
        "-dset",
        "--dataset",
        dest="filename",
        type=str,
        help="The path to the BIDS dataset containing fMRI "
        "data and fMRIPrep derivatives",
        required=True,
    )
    optional.add_argument(
        "-info",
        "--info",
        dest="info",
        action="store_true",
        help="Only output info about the file, don't process. "
        "Default is to process.",
        default=False,
    )
    optional.add_argument(
        "-indir",
        "--input-dir",
        dest="indir",
        type=str,
        help="Folder containing input. " "Default is current folder.",
        default=".",
    )
    optional.add_argument(
        "-outdir",
        "--output-dir",
        dest="outdir",
        type=str,
        help="Folder where output should be placed. "
        "Default is current folder. "
        'If "-heur" is used, it\'ll become '
        'the site folder. Requires "-sub". '
        'Optional to specify "-ses".',
        default=".",
    )
    optional.add_argument(
        "-heur",
        "--heuristic",
        dest="heur_file",
        type=str,
        help="File containing heuristic, with or without "
        "extension. This file is needed in order to "
        "convert your input file to BIDS format! "
        "If no path is specified, it assumes the file is "
        "in the current folder. Edit the heur_ex.py file in "
        "heuristics folder.",
        default=None,
    )
    optional.add_argument(
        "-sub",
        "--subject",
        dest="sub",
        type=str,
        help='Specify alongside "-heur". Code of ' "subject to process.",
        default=None,
    )
    optional.add_argument(
        "-ses",
        "--session",
        dest="ses",
        type=str,
        help='Specify alongside "-heur". Code of ' "session to process.",
        default=None,
    )
    optional.add_argument(
        "-chtrig",
        "--channel-trigger",
        dest="chtrig",
        type=int,
        help="The column number of the trigger channel. "
        "Channel numbering starts with 1. "
        "Default is 0. If chtrig is left as zero phys2bids will "
        "perform an automatic trigger channel search by channel names.",
        default=0,
    )
    optional.add_argument(
        "-chsel",
        "--channel-selection",
        dest="chsel",
        nargs="*",
        type=int,
        help="The column numbers of  the channels to process. "
        "Default is to process all channels.",
        default=None,
    )
    optional.add_argument(
        "-ntp",
        "--numtps",
        dest="num_timepoints_expected",
        nargs="*",
        type=int,
        help="Number of expected trigger timepoints (TRs). "
        "Default is None. Note: the estimation of beggining of "
        "neuroimaging acquisition cannot take place with this default. "
        "If you're running phys2bids on a multi-run recording, "
        "give a list of each expected ntp for each run.",
        default=None,
    )
    optional.add_argument(
        "-tr",
        "--tr",
        dest="tr",
        nargs="*",
        type=float,
        help="TR of sequence in seconds. "
        "If you're running phys2bids on a multi-run recording, "
        "you can give a list of each expected ntp for each run, "
        "or just one TR if it is consistent throughout the session.",
        default=None,
    )
    optional.add_argument(
        "-thr",
        "--threshold",
        dest="thr",
        type=float,
        help="Threshold to use for trigger detection. "
        'If "ntp" and "TR" are specified, phys2bids '
        "automatically computes a threshold to detect "
        "the triggers. Use this parameter to set it manually. "
        "This parameter is necessary for multi-run recordings. ",
        default=None,
    )
    optional.add_argument(
        "-pad",
        "--padding",
        dest="pad",
        type=float,
        help="Padding in seconds used around a single run "
        "when separating multi-run session files. "
        "Default is 9 seconds.",
        default=9,
    )
    optional.add_argument(
        "-chnames",
        "--channel-names",
        dest="ch_name",
        nargs="*",
        type=str,
        help="Column header (for json file output).",
        default=[],
    )
    optional.add_argument(
        "-yml",
        "--participant-yml",
        dest="yml",
        type=str,
        help="full path to file with info needed to generate " "participant.tsv file ",
        default="",
    )
    optional.add_argument(
        "-debug",
        "--debug",
        dest="debug",
        action="store_true",
        help="Only print debugging info to log file. Default is False.",
        default=False,
    )
    optional.add_argument(
        "-quiet",
        "--quiet",
        dest="quiet",
        action="store_true",
        help="Only print warnings to log file. Default is False.",
        default=False,
    )
    optional.add_argument(
        "-v", "--version", action="version", version=("%(prog)s " + __version__)
    )

    parser._action_groups.append(optional)

    return parser


def print_summary(filename, ntp_expected, ntp_found, samp_freq, time_offset, outfile):
    """
    Print a summary onscreen and in file with informations on the files.
    Parameters
    ----------
    dset: str
        Name of the input dataset of idconn.
    subjects: int
        Number of expected timepoints, as defined by user.
    ntp_found: int
        Number of timepoints found with the automatic process.
    samp_freq: float
        Frequency of sampling for the output file.
    time_offset: float
        Difference between beginning of file and first TR.
    outfile: str or path
        Fullpath to output file.
    Notes
    -----
    Outcome:
    summary: str
        Prints the summary on screen
    outfile: .log file
        File containing summary
    """
    start_time = -time_offset
    summary = (
        f"\n------------------------------------------------\n"
        f"Filename:            {filename}\n"
        f"\n"
        f"Timepoints expected: {ntp_expected}\n"
        f"Timepoints found:    {ntp_found}\n"
        f"Sampling Frequency:  {samp_freq} Hz\n"
        f"Sampling started at: {start_time:.4f} s\n"
        f"Tip: Time 0 is the time of first trigger\n"
        f"------------------------------------------------\n"
    )
    LGR.info(summary)
    utils.write_file(outfile, ".log", summary)


@due.dcite(
    Doi("10.1038/sdata.2016.44"),
    path="phys2bids",
    description="The BIDS specification",
    cite_module=True,
)
def idconn(
    filename,
    info=False,
    indir=".",
    outdir=".",
    heur_file=None,
    sub=None,
    ses=None,
    chtrig=0,
    chsel=None,
    num_timepoints_expected=None,
    tr=None,
    thr=None,
    pad=9,
    ch_name=[],
    yml="",
    debug=False,
    quiet=False,
):
    """
    Run main workflow of IDConn.
    Runs the parser, builds statsmodel json, then reads in fmri data + mask + task timing,
    makes and saves connectivity matrices, computes graph measures, and imputs missing data.
    If only info is required,
    it returns a summary onscreen.
    Otherwise, it operates on the input to return a .tsv.gz file, possibly
    in BIDS format.
    Raises
    ------
    NotImplementedError
        If the file extension is not supported yet.
    """
    # Check options to make them internally coherent pt. I
    # #!# This can probably be done while parsing?
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    logdir = os.path.join(outdir, "logs")
    os.makedirs(logdir, exist_ok=True)

    # Create logfile name
    basename = "idconn_"
    extension = "tsv"
    isotime = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    logname = os.path.join(logdir, (basename + isotime + "." + extension))

    # Set logging format
    log_formatter = logging.Formatter(
        "%(asctime)s\t%(name)-12s\t%(levelname)-8s\t%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Set up logging file and open it for writing
    log_handler = logging.FileHandler(logname)
    log_handler.setFormatter(log_formatter)
    sh = logging.StreamHandler()

    if quiet:
        logging.basicConfig(
            level=logging.WARNING,
            handlers=[log_handler, sh],
            format="%(levelname)-10s %(message)s",
        )
    elif debug:
        logging.basicConfig(
            level=logging.DEBUG,
            handlers=[log_handler, sh],
            format="%(levelname)-10s %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            handlers=[log_handler, sh],
            format="%(levelname)-10s %(message)s",
        )

    version_number = _version.get_versions()["version"]
    LGR.info(f"Currently running IDConn version {version_number}")
    LGR.info(f"BIDS dset derivatives live at {deriv_dir}")

    # Save call.sh
    arg_str = " ".join(sys.argv[1:])
    call_str = f"idconn {arg_str}"
    f = open(os.path.join(logdir, "call.sh"), "a")
    f.write(f"#!bin/bash \n{call_str}")
    f.close()

    ###########
    # Need parser to include name and desc of model
    ###########
    statsmodels_path = os.path.join(outdir, "model-{name}_desc-{desc}_smdl.json")
    LGR.info(f"Creating BIDS Stats Models json @ {statsmodels_path}")
    model = io.build_statsmodel_json(
        name,
        task,
        contrast,
        confounds,
        highpass,
        mask,
        conn_meas,
        graph_meas,
        exclude=None,
        outfile=statsmodels_path,
    )

    # How do I get subjects from the model? Use pybids!

    ###########
    # Need parser to include space and task name
    ###########
    assert exists(
        dset_dir
    ), "Specified dataset doesn't exist:\n{dset_dir} not found.\n\nPlease check the filepath."
    layout = bids.BIDSLayout(dset_dir, derivatives=True)
    subjects = layout.get(return_type="id", target="subject", suffix="bold")
    sessions = layout.get(return_type="id", target="session", suffix="bold")
    runs = layout.get(return_type="id", target="session", suffix="bold")
    preproc_subjects = layout2.get(
        scope="fmriprep",
        return_type="id",
        target="subject",
        task=task,
        space=space,
        desc="preproc",
        suffix="bold",
    )
    if len(subjects) != len(preproc_subjects):
        LGR.info(
            f"{len(subjects)} subjects found in dset, only {len(preproc_subjects)} have preprocessed BOLD data. Pipeline is contniuing anyway, please double check preprocessed data if this doesn't seem right."
        )

    LGR.info(f"Computing connectivity matrices using {atlas}")
    for subject in subjects:
        LGR.info(f"Subject {subject}")
        for session in sessions:
            LGR.info(f"Session {session}")
            adj_matrix = estimate_connectivity(
                layout,
                subject,
                session,
                runs,
                connectivity_metric,
                space,
                atlas,
                confounds,
            )
            # if graph_measures:
            # for measure in graph_measures:
            # estimate_thresh
            # for threshold in bounds


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
