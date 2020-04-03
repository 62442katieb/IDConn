#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import bct
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from nilearn import plotting
from os.path import join
from datetime import datetime

# In[2]:


data_dir = '/Users/katherine/Dropbox/Projects/physics-retrieval/data/output'
fig_dir = '/Users/katherine/Dropbox/Projects/physics-retrieval/figures'
shen_nii = '/Users/katherine/Dropbox/Projects/physics-retrieval/shen2015_2mm_268_parcellation.nii.gz'


# In[3]:


def grab_corr(subjects, nodes, task, condition, session, atlas, verbose=False):
    now = datetime.now()
    print(task, condition, session, now.strftime("%H:%M:%S"))
    errors = pd.Series(index=subjects)
    corrmats = []
    for subject in subjects:
        try:
            if condition != None:
                corrmat = np.genfromtxt(join(data_dir, '{0}-session-{1}_{2}-{3}_{4}-corrmat.csv'.format(subject, 
                                                                                                        session, 
                                                                                                        task, 
                                                                                                        condition, 
                                                                                                        atlas)),
                                        delimiter=' ')
            else:
                corrmat = np.genfromtxt(join(data_dir, '{0}-session-{1}-{2}_network_corrmat_{3}.csv'.format(subject, 
                                                                                                        session, 
                                                                                                        task, 
                                                                                                        atlas)),
                                        delimiter=',')
            corrmats.append(corrmat)
        except Exception as e:
            if verbose:
                print(subject, session, task, condition, atlas, 'error.')
            else:
                pass
            errors[subject] = e
    return corrmats, errors


# In[4]:


def nbs_and_graphs(corr1, corr2, p_thresh, k, atlas, verbose):
    coordinates = plotting.find_parcellation_cut_coords(labels_img=atlas)
    corr1 = np.asarray(corr1, dtype='float')
    corr2 = np.asarray(corr2, dtype='float')

    if corr1.shape[0] != corr1.shape[1]:
        corr1 = np.moveaxis(corr1, 0, -1)
        corr2 = np.moveaxis(corr2, 0, -1)
    
    thresh = stats.t.isf(p_thresh, corr1.shape[2])
    pval, adj, _ = bct.nbs_bct(corr1,
                               corr2,
                               thresh,
                               k=k,
                               tail='both',
                               paired=True,
                               verbose=verbose)
    print(pval)
    gridkw = dict(width_ratios=[1,2])
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw=gridkw, figsize=(15,4))

    g = sns.heatmap(adj, square=True, ax=ax1, cmap='Greys')
    h = plotting.plot_connectome_strength(adj, node_coords=coordinates, cmap='YlGnBu', axes=ax2)

    return pval, adj, fig


# In[5]:


bx_dir = '/Users/katherine/Dropbox/Projects/physics-retrieval/data/rescored'
b_df = pd.read_csv(join(bx_dir, 'non-brain-data.csv'), index_col=0, header=0)

b_df.drop(['GPA', 'Age', 'Handedness', 'Strt.Level', 'RetrPhyAcc1',
       'Mean Correct RT Pre', 'RetrPhyAcc2', 'Mean Correct RT Post',
       'FCIPhyAcc1', 'FCIPhyAcc2', 'GID Pre', 'GID Post',
       'deltaRetrPhyAcc', 'deltaFCIPhyAcc', 'Phy48Grade', 'Sex', 'Ethnic.Grp',
       'Class.Type', 'Subject', 'Lec', 'SexXClass', 'VCI1XClass',
       'VCI2XClass', 'deltaVCIXClass', 'PRI1XClass', 'PRI2XClass',
       'deltaPRIXClass', 'WMI1XClass', 'WMI2XClass', 'deltaWMIXClass',
       'PSI1XClass', 'PSI2XClass', 'deltaPSIXClass', 'FSIQ1XClass',
       'FSIQ2XClass', 'deltaFSIQXClass', 'VCI1XClassXSex', 'VCI1XSex',
       'VCI2XClassXSex', 'VCI2XSex', 'deltaVCIXClassXSex', 'deltaVCIXSex',
       'PRI1XClassXSex', 'PRI1XSex', 'PRI2XClassXSex', 'PRI2XSex',
       'deltaPRIXClassXSex', 'deltaPRIXSex', 'WMI1XClassXSex', 'WMI1XSex',
       'WMI2XClassXSex', 'WMI2XSex', 'deltaWMIXClassXSex', 'deltaWMIXSex',
       'PSI1XClassXSex', 'PSI1XSex', 'PSI2XClassXSex', 'PSI2XSex',
       'deltaPSIXClassXSex', 'deltaPSIXSex', 'FSIQ1XClassXSex', 'FSIQ1XSex',
       'FSIQ2XClassXSex', 'FSIQ2XSex', 'deltaFSIQXClassXSex', 'deltaFSIQXSex'], axis=1, inplace=True)


# In[6]:


husl_pal = sns.husl_palette(h=0, n_colors=268)
crayons_l = sns.crayon_palette(['Vivid Tangerine', 'Cornflower'])
crayons_d = sns.crayon_palette(['Brick Red', 'Midnight Blue'])
grays = sns.light_palette('#999999', n_colors=3, reverse=True)

f_2 = sns.crayon_palette(['Red Orange', 'Vivid Tangerine'])
m_2 = sns.crayon_palette(['Cornflower', 'Cerulean'])


# ## Physics knowledge task
# 1. All students: physics > general
# 2. Female students: physics > general
#     1. All
#     2. Active learning
#     3. Traditional lecture
# 3. Male students: physics > general
#     1. All
#     2. Active learning
#     3. Traditional lecture
# 4. Active learning: physics > general
# 5. Traditional lecture: physics > general


f_subs = b_df[b_df['F'] == 1].index
m_subs = b_df[b_df['F'] == 0].index
a_subs = b_df[b_df['Mod'] == 1].index
l_subs = b_df[b_df['Mod'] == 0].index

fmod = b_df[b_df['F'] == 1]
af = fmod[fmod['Mod'] == 1]
af_subs = af.index

flec = b_df[b_df['F'] == 1]
lf = fmod[fmod['Mod'] == 0]
lf_subs = lf.index

mmod = b_df[b_df['F'] == 0]
am = mmod[mmod['Mod'] == 1]
am_subs = am.index

mlec = b_df[b_df['F'] == 0]
lm = mlec[mlec['Mod'] == 0]
lm_subs = lm.index

subject_groups = {
                  #'all': b_df.index,
                  #'female': f_subs, 
                  #'male': m_subs,
                  #'lecture': l_subs, 
                  #'modeling': a_subs,
                  'female_lecture': lf_subs, 
                  'female_modeling': af_subs,
                  'male_lecture': lm_subs, 
                  'male_modeling': am_subs}

tasks = {'retr': ['Physics', 'General'], 
         'fci': ['Physics', 'NonPhysics']}
for task in tasks.keys():
    for group in subject_groups.keys():
        print(group)
        if group == 'all':
            print('pass')
        if group == 'female':
            print('pass')
        if group == 'male':
            print('pass')
        if group == 'lecture':
            print('pass')
        if group == 'modeling':
            print('pass')
        else:
            print(task, group)
            subjects = subject_groups[group]
            cond0 = tasks[task][0]
            cond1 = tasks[task][1]

            cond_c, cond_e = grab_corr(subjects,
                                    nodes=None, 
                                    task=task, 
                                    condition=cond0,
                                    session='1',
                                    atlas='shen2015')

            ctrl_c, ctrl_e = grab_corr(subjects,
                                    nodes=None,
                                    task=task,
                                    condition=cond1,
                                    session='1',
                                    atlas='shen2015')

            p, adj, fig = nbs_and_graphs(ctrl_c, cond_c, p_thresh=0.05 , k=1000, atlas=shen_nii, verbose=False)

            centrality = bct.betweenness_bin(adj)
            centrality_df = pd.Series(centrality, index=np.arange(1,269), name='betweenness centrality')
            centrality_df.to_csv(join(data_dir, 'nbs', '{0}_students-{1}-centrality.csv'.format(group, task)), 
                                header=False)

            adjacency = pd.DataFrame(adj, columns=np.arange(1,269), index=np.arange(1,269))
            adjacency.to_csv(join(data_dir, 'nbs', '{0}_students-{1}.csv'.format(group, task)))

            fig.savefig(join(fig_dir, 'nbs-{0}_students-{1}.png'.format(group, task)), dpi=300)

