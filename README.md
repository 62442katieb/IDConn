# IDConn
Pipeline for studying individual differences in network connectivity and organization in resting-state or task fMRI data

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/134580580.svg)](https://zenodo.org/badge/latestdoi/134580580)


IDConn weaves together several tools from existing software packages to create a unified workflow for assessing individual differences in functional brain connectivity either at rest or during a task.
<br><br>
The workflow's steps are arranged in modules that each serve a distinct purpose in the data analysis stream.
1. `connectivity` includes the functionality necessary to create network connectivity matrices from fMRI data.
2. `networking` includes tools for thresholding graphs, calculating network measures, and creating null distributions.
3. `data` includes tools for describing whole datasets, diagnosing and treating missing data, and orthogonalizing data as necessary.
4. `statistics` includes tools necessary for performing inferential statistics from processed data.
5. `figures` includes tools for describing data and results graphically, from brain images to scatterplots to heatmaps.
![](./docs/logo/IDConnWorkflowH.svg)

