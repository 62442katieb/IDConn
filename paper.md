---
title: 'IDConn: Individual differences in functional brain connectivity'
tags:
  - Python
  - fMRI
  - functional connectivity
  - human neuroimaging
  - 
authors:
  - name: Katherine L. Bottenhorn
    orcid: 0000-0002-7796-8795
    affiliation: 1
  - name: Matthew T. Sutherland
    affiliation: 1
  - name: Angela R. Laird
    affiliation: "1, 2"
affiliations:
 - name: Department of Psychology, Florida International University, Miami, FL, USA
   index: 1
 - name: Department of Physics, Florida International University, Miami, FL, USA
   index: 2
 - name: Center for Imaging Science, Florida International University, Miami, FL, USA
   index: 3
date: 02 October 2020
bibliography: paper.bib

---

# Summary

Historically, human neuroimaging in cognitive neuroscience has focused on understanding 
which brain regions are “active” during different sorts of thinking and behavior and how 
this differs across groups. While this approach has led to advances in our understanding 
of brain function and functional anatomy, as well as the neurobiology of disease etiology, 
it has two distinct shortcomings. First, focusing on brain activation substantially limits 
our ability to understand the brain, as it ignores interactions between brain regions and 
behind-the scenes brain activity vital to cognition and behavior. In the past decade, the 
focus has shifted to include functional brain connectivity, which estimates communication 
between spatially distinct brain regions, and network-based measures of brain organization 
that are based on this connectivity. Such research has uncovered large-scale brain networks 
that are coherent across experimental paradigms and behaviors, but which exhibit subtle, yet 
consistent, differences across development, aging, behavior, and both psychiatric and 
neurological diagnoses. Without the constraints inherent in activation-focused research, 
connectivity-focused neuroimaging research is able to more comprehensively assess neurobiology 
during experimental paradigms. Second, attempting to view the neurobiology of behavior, 
development, and disease through the lens of group differences at worst artificially imposes 
false dichotomies and at best ignores the wealth of information in within-group heterogeneity. 
Recent research has shown that the relative magnitudes of variability in brain network 
organization across a group and within each individual are commensurate. Together, these 
phenomena illustrate a greater need for understanding sources of individual variability in 
brain network organization and how it relates to behavior. IDConn is a data 
analysis workflow for combining methods for assessing <ins>i</ins>ndividual 
<ins>d</ins>ifferences in brain <ins>conn</ins>ectivity and organization from functional 
magnetic resonance imaging (fMRI) data with robust statistical methods for assessing 
relationships between continuous variables, adjusted to address the unique challenges of 
fMRI data.

# Statement of need 

`IDConn` is a configurable data analysis pipeline written in Python for 
assessing individual differences in functional connectivity and derived
network measures. It brings together existing tools for neuroimaging data
analysis, data science, machine learning, graph theoretic network analysis, 
statistics, and scientific computing into a unified, streamlined workflow
for sophisticated, rigorous computation and extraction of connectivity and 
graph theoretic measures, for linear regression-based analyses with behavioral,
demographic, psychophysiological, or other continuous variables. Furthermore, 
it conforms with the Brain Imaging Data Structure (BIDS; [@Gorgolewski:2016])
for optimal redistribution and sharing of pipeline outputs.

`IDConn` was designed for applications in human neuroimaging research, providing
a flexible, open data analysis stream that takes in preprocessed fMRI data and 
provdes computed graphs, derived graph measures, statistical models and the results
thereof, and, optionally, figures presenting these results, all in an organized,
sharing-friendly format for optimal reproducibility and transparency. It has already 
been used in a number of scientific publications [@Gonzalez&Bottenhorn:2019; 
@Bottenhorn:2020]. Bringing together robust tools for fMRI data processing, 
dataset cleaning, and statistical inference, `IDConn` will enable researchers to 
perform statistically-robust assessments of individual differences in functional
brain connectivity and organization by neuroimaging researchers from the Python-naiive 
to developers.

# State of the Field

Currently, there are a wealth of software tools available for analyzing neuroimaging data.
Tools such as the FMRIB Software Library ([FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki); 
@Jenkinson:2012; @Smith:2004; @Woolrich:2009), Analysis of Functional NeuroImages 
([AFNI](http://afni.nimh.nih.gov/); @Cox:1996), Statistical Parametric Mapping 
([SPM](); @Frackowiak:2004), and [Brain Voyager](https://www.brainvoyager.com/index.html) 
[@Goebel:2012] provide graphical user interfaces to aid researchers in the analysis of fMRI 
data, along with scripting options to automate and batch-process analyses via command-line 
interfaces. However, the code bases of these tools have varying degrees of accessibility and 
some are completely closed. Open source tools for fMRI data analysis exist, too, including
[Nilearn](https://nilearn.github.io/) [@Abraham:2014] and [NiPy](http://nipy.org/nipy/) 
[@Millman:2007]. The latter is a tool as well as a community, under whose umbrella fall Nilearn
as well as [Nipype](https://nipype.readthedocs.io/en/latest/) [@Gorgolewski:2011]. Nipype is an
open source data processing framework that provides interfaces for a host of neuroimaging data
analysis tools, implementations of a number of analysis algorithms, and a pipeline engine for 
connecting tools into unified workflows in Python. It allows researchers to integrate tools from 
existing, and perhaps less "open", software packages in transparent, reproducible, and reusable
Python scripts. 

`IDConn` has a similar goal, though with a narrower scope and more specific 
applications. Instead of a basis for building data analysis workflows, `IDConn` brings together 
existing tools to create a configurable pipeline that is accessible to researchers, regardless 
of their technical background. It integrates functions from Nilearn, Nipype, scikit-learn, SciPy,
Numpy, [NiBabel](https://nipy.org/nibabel/), Pandas, Seaborn, statsmodels, NetworkX, the Brain 
Connectivity Toolbox in Python ([bct](https://github.com/aestrivex/bctpy); @Rubinov:2010), 
Neurosynth, and NiMARE to provide \autoref{fig:wfdiagram}:
1. Computation of brain connectivity graphs (including input/output support)
2. Computation of topological and graph theoretic measures from these graphs
3. Data organization, harmonization, and treatment of missing data
4. Statistical analysis, including multivariate regressions and robust corrections for multiple 
comparisons
5. Optional automated generation of publication-quality figures based on these analyses
6. Optional quantitative functional decoding of brain region and network based results
![IDConn takes in resting-state or task-based fMRI data and maps of brain regions and/or networks 
of interest to compute individual measures of connectivity and/or topology and assesses their 
relation with individual demographic, behavioral, and other measures to provide robust statistical 
inference of individual differences in the brain that relate to behavior, etc., and figures 
thereof.\label{fig:wfdiagram}](figure1.png)

Finally, `IDConn` has built-in recommended "best practices" for 
- multiple comparisons corrections
- multi-threshold estimation of graph theoretic properties
- task-based graph analyses
- nuisance regression
in addition to thorough documentation of how researchers should set up analyses. Altogether, IDConn
makes it easy for researchers to perform robust analyses of brain connectivity, regardless of their 
technical backgrounds and bridges the gap between development-minded neuroimaging researchers and 
code-naiive researchers.

# Acknowledgements

This work would not be possible without the support of numerous members of the Nipype community, 
BIDS working groups, and Cognitive Neuroscience PhD program at Florida International University.

# References
