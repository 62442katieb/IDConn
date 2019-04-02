
# coding: utf-8

# In[4]:


from __future__ import division
from os.path import join, basename, exists
from os import makedirs
from glob import glob

from nilearn import input_data, datasets, plotting, regions
from nilearn.image import concat_imgs
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from scipy.stats import pearsonr

import bct
import json
import numpy as np
import pandas as pd


# ## Preprocessing
# Largely following the Westphal et al. (2017) paper, but taking into account the things that Dani Bassett does in her papers (which I still need to look into).
# ### Preprocessing methods per Westphal et al., 2017
# Whole-brain MRI was administered on a 3.0 Tesla Siemens TIM Trio scanner at the UCLA Staglin Center for Cognitive Neuroscience. Functional images were ac- quired using a T2*-weighted echoplanar imaging sequence (TR   2.0 s; TE   30 ms; flip angle   75°; FOV   19.2 cm; voxel resolution   3.0   3.0   3.7 mm; 33 interleaved axial slices). The first three volumes of each 239-volume run were discarded to ensure T1 stabilization.<br>Preprocessing was done in SPM8.
# 1. Slice timing correction N
# 2. Motion correction Y
# 3. Unwarping N
# 4. Coregistration to subject's T1 Y
# 5. Anatomical segmentation N
# 6. Spatial normalization to MNI template REVERSE
# 7. Spatial smoothing (6mm FWHM) N
# 8. High-pass filtering (236_s_) Y
# 9. Timecourse per voxel demeaned. Y
# ### Alterations made below
# Preprocessing was done with FSL tools in Nipype.
# 3. No fieldmaps, so no unwarping... (look into this)
# 7. No smoothing
# 8. High pass filtering at 55s
# 9. Standardized TS

# In[1]:





#choose your atlas and either fetch it from Nilearn using one of the the 'datasets' functions
#laird_2011_icns = '/home/data/nbc/physics-learning/retrieval-graphtheory/18-networks-5.14-mni_2mm.nii.gz'
#laird_2011_icns = '/Users/Katie/Dropbox/Projects/physics-retrieval/18-networks-5.14.nii.gz'
#add labels for the parcellation
labels = ['limbic', 'orbitofrontal', 'basal ganglia', 'salience', 'hunger',
          'motor learning', 'frontoparietal', 'hand', 'motor execution', 'higher order visual',
          'lateral visual', 'medial visual', 'default mode',' cerebellum', 'right central executive',
          'auditory', 'mouth', 'left central executive']

#harvox_hippo = '/home/data/nbc/physics-learning/retrieval-graphtheory/harvox-hippo-prob50-2mm.nii.gz'
#masks = [laird_2011_icns, harvox_hippo]
#mask_names = ['18_icn', 'hippo']


# In[ ]:


#only want post subjects
subjects = ['101', '102', '103', '104', '106', '107', '108', '110', '212',
            '214', '215', '216', '217', '218', '219', '320', '321', '323',
            '324', '325', '327', '328', '330', '331', '333', '334',
            '335', '336', '337', '338', '339', '340', '341', '342', '343', '344',
            '345', '346', '347', '348', '349', '350', '451', '453', '455',
            '458', '459', '460', '462', '463', '464', '465', '467',
            '468', '469', '470', '502', '503', '571', '572', '573', '574',
            '577', '578', '581', '582', '584', '585', '586', '587',
            '588', '589', '591', '592', '593', '594', '595', '596', '597',
            '598', '604', '605', '606', '607', '608', '609', '610', '612',
            '613', '614', '615', '617', '618', '619', '620', '621', '622',
            '623', '624', '625', '626', '627', '629', '630', '631', '633',
            '634']
#all subjects 102 103 101 104 106 107 108 110 212 X213 214 215 216 217 218 219 320 321 X322 323 324 325
#327 328 X329 330 331 X332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 451
#X452 453 455 X456 X457 458 459 460 462 463 464 465 467 468 469 470 502 503 571 572 573 574 X575 577 578
#X579 X580 581 582 584 585 586 587 588 589 X590 591 592 593 594 595 596 597 598 604 605 606 607 608 609
#610 X611 612 613 614 615 X616 617 618 619 620 621 622 623 624 625 626 627 X628 629 630 631 633 634
#errors in fnirt-to-mni: 213, 322, 329, 332, 452, 456, 457, 575, 579, 580, 590, 611, 616, 628
#subjects without post-IQ measure: 452, 461, 501, 575, 576, 579, 583, 611, 616, 628, 105, 109, 211, 213, 322, 326, 329, 332
#subjects = ['101','103']


# In[5]:


#data_dir = '/home/data/nbc/physics-learning/data/pre-processed'
data_dir = '/home/data/nbc/physics-learning/retrieval-graphtheory/output'
#sink_dir = '/Users/Katie/Dropbox/Projects/physics-retrieval/data/out'

runs = [0, 1]
connectivity_metric = 'correlation'
networks = [12, 14, 17]
conditions = ['phy', 'gen']
thresh_range = np.arange(0.1, 1, 0.1)
highpass = 1/55.

correlation_measure = ConnectivityMeasure(kind=connectivity_metric)


# In[ ]:


#gen_timing = np.genfromtxt('/home/data/nbc/physics-learning/physics-learning/RETRconditionGeneralSess1.txt',
#                           delimiter='\t')
gen_timing = np.genfromtxt('/home/data/nbc/physics-learning/retrieval-graphtheory/RETRconditionGeneralSess1.txt',
                           delimiter='\t', dtype=int)

gen_timing = (gen_timing/2)-1
gen_timing = gen_timing[:,0:2]

#phy_timing = np.genfromtxt('/home/data/nbc/physics-learning/physics-learning/RETRconditionPhysicsSess1.txt',
#                           delimiter='\t')
phy_timing = np.genfromtxt('/home/data/nbc/physics-learning/retrieval-graphtheory/RETRconditionPhysicsSess1.txt',
                           delimiter='\t')
phy_timing = (phy_timing/2)-1
phy_timing = phy_timing[:,0:2]
timing = {}
timing['phy'] = phy_timing
timing['gen'] = gen_timing


# In[ ]:


df = pd.DataFrame({'global efficiency phy': np.empty([len(subjects),], dtype=float),
                   'fc default mode-left central executive phy': np.empty([len(subjects),], dtype=float),
                   'fc default mode-right central executive phy': np.empty([len(subjects),], dtype=float),
                   'fc left central executive-right central executive phy': np.empty([len(subjects),], dtype=float),
                   'fc hippo-default mode phy': np.empty([len(subjects),], dtype=float),
                   'fc hippo-left central executive phy': np.empty([len(subjects),], dtype=float),
                   'fc hippo-right central executive phy': np.empty([len(subjects),], dtype=float),
                   'le default mode phy': np.empty([len(subjects),], dtype=float),
                   'le right central executive phy': np.empty([len(subjects),], dtype=float),
                   'le left central executive phy': np.empty([len(subjects),], dtype=float),
                   'global efficiency gen': np.empty([len(subjects),], dtype=float),
                   'fc default mode-left central executive gen': np.empty([len(subjects),], dtype=float),
                   'fc default mode-right central executive gen': np.empty([len(subjects),], dtype=float),
                   'fc left central executive-right central executive gen': np.empty([len(subjects),], dtype=float),
                   'fc hippo-default mode gen': np.empty([len(subjects),], dtype=float),
                   'fc hippo-left central executive gen': np.empty([len(subjects),], dtype=float),
                   'fc hippo-right central executive gen': np.empty([len(subjects),], dtype=float),
                   'le default mode gen': np.empty([len(subjects),], dtype=float),
                   'le right central executive gen': np.empty([len(subjects),], dtype=float),
                   'le left central executive gen': np.empty([len(subjects),], dtype=float)},
                  index=subjects)


#run preprocessing once per run per subject
#and maybe try regressing out the task to see if the results change with the remaining intrinsic FC
for subject in subjects:
    ntwk_run_cond = {}
    ntwk = {}
    hipp = {}
    hipp_run_cond = {}
    corrmats = {}
    for run in runs:
        #xfm laird 2011 maps to subject's epi space & define masker
        epi = join(data_dir, subject,'{0}-{1}_retr-mcf.nii.gz'.format(subject, run))
        confounds = join(data_dir, subject,'{0}-{1}_retr-confounds.txt'.format(subject, run))
        icn = join(data_dir, subject,'{0}-{1}_18_icn_retr.nii.gz'.format(subject, run))
        hippo = join(data_dir, subject,'{0}-{1}_hippo_retr.nii.gz'.format(subject, run))
        ntwk_masker = NiftiLabelsMasker(icn, standardize=True, high_pass=highpass, t_r=2., verbose=1)
        hipp_masker = NiftiLabelsMasker(hippo, standardize=True, high_pass=highpass, t_r=2., verbose=1)

        #extract the network-wise and hippocampus timeseries per run
        #fmri = join(data_dir, subject, 'session-1', 'retr', 'mni', '{0}_filtered_func_data_{1}.nii.gz'.format(subject, run))
        ntwk_ts = ntwk_masker.fit_transform(epi, confounds=confounds)
        hipp_ts = hipp_masker.fit_transform(epi, confounds=confounds)
        #ts = [ntwk_ts, hipp_ts]
        #and then separate each run's timeseries into the different conditions
        for condition in conditions:
            ntwk_run_cond['{0} {1}'.format(condition, run)] = np.vstack((ntwk_ts[timing[condition][0,0].astype(int):(timing[condition][0,0]+timing[condition][0,1]+1).astype(int), :], ntwk_ts[timing[condition][1,0].astype(int):(timing[condition][1,0]+timing[condition][1,1]+1).astype(int), :], ntwk_ts[timing[condition][2,0].astype(int):(timing[condition][2,0]+timing[condition][2,1]+1).astype(int), :]))
            hipp_run_cond['{0} {1}'.format(condition, run)] = np.vstack((hipp_ts[timing[condition][0,0].astype(int):(timing[condition][0,0]+timing[condition][0,1]+1).astype(int)], hipp_ts[timing[condition][1,0].astype(int):(timing[condition][1,0]+timing[condition][1,1]+1).astype(int)], hipp_ts[timing[condition][2,0].astype(int):(timing[condition][2,0]+timing[condition][2,1]+1).astype(int)]))
    for condition in conditions:
        ntwk[condition] = np.vstack((ntwk_run_cond['{0} 0'.format(condition)], ntwk_run_cond['{0} 1'.format(condition)]))
        hipp[condition] = np.vstack((hipp_run_cond['{0} 0'.format(condition)], hipp_run_cond['{0} 1'.format(condition)]))
        corrmats[condition] = correlation_measure.fit_transform([ntwk[condition]])[0]
        np.savetxt(join(data_dir, subject, '{0}-{1}-corrmat.csv'.format(subject, condition)), corrmats[condition])
        df.at[subject, 'fc default mode-right central executive {0}'.format(condition)] = corrmats[condition][12,14]
        df.at[subject, 'fc default mode-left central executive {0}'.format(condition)] = corrmats[condition][12,17]
        df.at[subject, 'fc left central executive-right central executive {0}'.format(condition)] = corrmats[condition][14,17]
        ge = []
        le = {}
        loceff = {}
        loceff['default mode'] = []
        loceff['left central executive'] = []
        loceff['right central executive'] = []
        for p in thresh_range:
            corrmat_thresh = bct.threshold_proportional(corrmats[condition], p, copy=True)
            #measures of interest here
            #global efficiency
            geff = bct.efficiency_wei(corrmat_thresh)
            ge.append(geff)

            #local efficiency
            leff = bct.efficiency_wei(corrmat_thresh, local=True)
            #print leff[2]
            for network in networks:
                #print network
                loceff[labels[network]].append(leff[network])
                #loceff['{0}, {1}'.format(labels[network], condition)].append(leff[network])
            #print loceff
            le['{0}, {1}'.format(p, condition)] = loceff

        #print 'global efficiency is {0}'.format(ge)
        df.at[subject, 'global efficiency {0}'.format(condition)] = np.trapz(ge, dx=0.1)

        #populate df with hippo-FCs and local efficiencies per network
        for network in networks:
            df.at[subject, 'le {0} {1}'.format(labels[network], condition)] = np.trapz(loceff[labels[network]], dx=0.1)
            #calculate hippocampus-network correlations (FC) per network & populate df
            network_ts = np.reshape(ntwk[condition][:,network], (84,1))
            #print(network_ts.shape)
            df.at[subject, 'fc hippo-{0} {1}'.format(labels[network], condition)] = pearsonr(hipp[condition], network_ts)[0]

df.to_csv(join(data_dir, 'out.csv'), sep=',')


# In[1]:


#for subject in subjects:
#    #print subject
#    if not exists(join(sink_dir, subject)):
#            makedirs(join(sink_dir, subject))
#    timeseries = {}
#    corrmats = {}
#    hippo = {}
#
#    #I SHOULD SWAP RUN AND CONDITION AND PUT PREPROC OUTSIDE OF CONDITION BECAUSE IT'S USELESS TO DUPLICATE THAT STEP
    #AND COMPUTATIONALLY INTENSIVE. PREPROC EACH RUN ONLY ONCE, GRAB THE TS, SPLICE CONDITION
#    for condition in conditions:
#        for i in np.arange(0, len(runs)):
#            #extract network-wise timeseries from nifti into array
#
#
#            #separate into different conditions
#            timeseries['{0} {1}'.format(condition, i)] = np.vstack((ts[timing[condition][0,0].astype(int):(timing[condition][0,0]+timing[condition][0,1]+1).astype(int), :], ts[timing[condition][1,0].astype(int):(timing[condition][1,0]+timing[condition][1,1]+1).astype(int), :], ts[timing[condition][2,0].astype(int):(timing[condition][2,0]+timing[condition][2,1]+1).astype(int), :]))

            #FOR SOME REASON, THIS CAUSES THEM TO BE HSTACKED? IDK WHAT'S GOING ON
#            hippo['{0} {1}'.format(condition, i)] = np.vstack((hipp_ts[timing[condition][0,0].astype(int):(timing[condition][0,0]+timing[condition][0,1]+1).astype(int)], hipp_ts[timing[condition][1,0].astype(int):(timing[condition][1,0]+timing[condition][1,1]+1).astype(int)], hipp_ts[timing[condition][2,0].astype(int):(timing[condition][2,0]+timing[condition][2,1]+1).astype(int)]))
            #print 'ts has shape {0}'.format(hippo['{0} {1}'.format(condition, i)].shape)
        #splice runs together
#        timeseries[condition] = np.vstack((timeseries['{0} 0'.format(condition)], timeseries['{0} 1'.format(condition)]))
        #print 'ts {0} = {1}'.format(condition, timeseries[condition].shape)

        #compute correlation matrices per condition per run
#        correlation_measure = ConnectivityMeasure(kind=connectivity_metric)
#        corrmats[condition] = correlation_measure.fit_transform([timeseries[condition]])[0]
#        np.savetxt(join(sink_dir, subject, '{0}-{1}-corrmat.csv'.format(subject, condition)), corrmats[condition])

        #populate df with network FC measures
#        hippo[condition] = np.vstack((hippo['{0} 0'.format(condition)], hippo['{0} 1'.format(condition)]))
        #print 'fc default mode-left central executive {0}'.format(condition), corrmats[condition][12,14]
#        df.at[subject, 'fc default mode-left central executive {0}'.format(condition)] = corrmats[condition][12,14]
#        df.at[subject, 'fc default mode-right central executive {0}'.format(condition)] = corrmats[condition][12,17]
#        df.at[subject, 'fc left central executive-right central executive {0}'.format(condition)] = corrmats[condition][14,17]

        #and since we're here, calculate graph theory measures & populate df
#        ge = []
#        le = {}
#        loceff = {}
#        loceff['default mode'] = []
#        loceff['left central executive'] = []
#        loceff['right central executive'] = []
#        for p in thresh_range:
#            corrmat_thresh = bct.threshold_proportional(corrmats[condition], p, copy=True)
            #measures of interest here
            #global efficiency
#            geff = bct.efficiency_wei(corrmat_thresh)
#            ge.append(geff)

            #local efficiency
#            leff = bct.efficiency_wei(corrmat_thresh, local=True)
#            #print leff[2]
#            for network in networks:
                #print network
#                loceff[labels[network]].append(leff[network])
                #loceff['{0}, {1}'.format(labels[network], condition)].append(leff[network])
            #print loceff
#            le['{0}, {1}'.format(p, condition)] = loceff

        #print 'global efficiency is {0}'.format(ge)
#        df.at[subject, 'global efficiency {0}'.format(condition)] = np.trapz(ge, dx=0.1)

        #populate df with hippo-FCs and local efficiencies per network
#        for network in networks:
#            df.at[subject, 'le {0} {1}'.format(labels[network], condition)] = np.trapz(loceff[labels[network]], dx=0.1)
            #calculate hippocampus-network correlations (FC) per network & populate df
#            network_ts = np.reshape(timeseries[condition][:,network], (84,1))
            #print(network_ts.shape)
#            df.at[subject, 'fc hippo-{0} {1}'.format(labels[network], condition)] = pearsonr(hippo[condition], network_ts)[0]
#df.to_csv(join(sink_dir, 'out.csv'), sep=',')


# In[ ]:


#local_dir = '/Users/Katie/Dropbox/Projects/physics-retrieval/data/out/'
phy = np.zeros((18, 18))
gen = np.zeros((18, 18))

for subject in subjects:
    phy += np.genfromtxt(join(data_dir, subject, '{0}-phy-corrmat.csv'.format(subject)), delimiter=' ')
    gen += np.genfromtxt(join(data_dir, subject, '{0}-gen-corrmat.csv'.format(subject)), delimiter=' ')
phy_corrmat = phy / len(subjects)
gen_corrmat = gen / len(subjects)

dif_corrmat = phy_corrmat - gen_corrmat

phy_corr_df = pd.DataFrame(phy_corrmat, index=labels, columns=labels)
gen_corr_df = pd.DataFrame(gen_corrmat, index=labels, columns=labels)
dif_corr_df = pd.DataFrame(dif_corrmat, index=labels, columns=labels)

#np.savetxt(join(local_dir, 'avg_phys_corrmat.csv'), phy_corrmat, delimiter=',')
#np.savetxt(join(local_dir, 'avg_gen_corrmat.csv'), gen_corrmat, delimiter=',')

phy_corr_df.to_csv(join(data_dir, 'avg_phys_corrmat.csv'), sep=',')
gen_corr_df.to_csv(join(data_dir, 'avg_gen_corrmat.csv'), sep=',')
dif_corr_df.to_csv(join(data_dir, 'avg_dif_corrmat.csv'), sep=',')


# def adj_to_list(input_filename,output_filename,delimiter):
#     import pandas as pd
#     import numpy as np
#     import networkx as nx
#     import csv
#     '''https://github.com/jermainkaminski/Adjacency-Matrix-to-Edge-List/blob/master/adjacencymatrix_to_edgelist.ipynb'''
#     '''Takes the adjacency matrix on file input_filename into a list of edges and saves it into output_filename'''
#     A=pd.read_csv(input_filename,delimiter=delimiter,index_col=0)
#     List=[('Source','Target','Weight')]
#     for source in A.index.values:
#         for target in A.index.values:
#             List.append((target,source,A[source][target]))
#     with open(output_filename, "wb") as f:
#         writer = csv.writer(f)
#         writer.writerows(List)
#     return List

# phy_list = adj_to_list(join(sink_dir, 'avg_phys_corrmat.csv'), join(sink_dir, 'avg_phys_edges.csv'), ',')
# gen_list = adj_to_list(join(sink_dir, 'avg_gen_corrmat.csv'), join(sink_dir, 'avg_gen_edges.csv'), ',')
# dif_list = adj_to_list(join(sink_dir, 'avg_dif_corrmat.csv'), join(sink_dir, 'avg_dif_edges.csv'), ',')
