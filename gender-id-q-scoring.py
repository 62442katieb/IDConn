from os.path import join
from glob import glob

import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style(style='whitegrid')
from scipy.stats import pearsonr
from scipy.stats import ttest_ind, ttest_rel


#less important for plotting

import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from statsmodels.sandbox.stats.multicomp import multipletests
from nilearn.mass_univariate import permuted_ols

data_dir = '/Users/Katie/Dropbox/Projects/physics-retrieval/data'
fig_dir = '/Users/Katie/Dropbox/Projects/physics-retrieval/figures'

df = pd.read_excel(join(data_dir, 'gender-identity-prepost.xlsx'), index_col=0, header=0)

m_df = df[df['Sex.1'] == 1]
m_df = m_df.drop(['Sex.1', 'GIF1.1', 'GIF2.1', 'GIF3.1', 'GIF4.1', 'GIF5.1', 'GIF6.1', 'GIF7.1', 'GIF8.1', 'GIF9.1', 'GIF10.1', 'GIF11.1', 'GIF12.1', 'GIF13.1', 'GIF14.1', 'GIF15.1', 'GIF16.1', 'GIF17.1', 'GIF18.1', 'GIF19.1', 'GIF1.2', 'GIF2.2', 'GIF3.2', 'GIF4.2', 'GIF5.2', 'GIF6.2', 'GIF7.2', 'GIF8.2', 'GIF9.2', 'GIF10.2', 'GIF11.2', 'GIF12.2', 'GIF13.2', 'GIF14.2', 'GIF15.2', 'GIF16.2', 'GIF17.2', 'GIF18.2','GIF19.2', 'GIM13.2'], axis=1)

f_df = df[df['Sex.1'] == 2]
f_df = f_df.drop(['Sex.1', 'GIM1.1', 'GIM2.1', 'GIM3.1', 'GIM4.1', 'GIM5.1', 'GIM6.1', 'GIM7.1', 'GIM8.1', 'GIM9.1', 'GIM10.1', 'GIM11.1', 'GIM12.1', 'GIM13.1', 'GIM14.1', 'GIM15.1', 'GIM16.1', 'GIM17.1', 'GIM18.1',  'GIM19.1', 'GIM1.2', 'GIM2.2', 'GIM3.2', 'GIM4.2', 'GIM5.2', 'GIM6.2', 'GIM7.2', 'GIM8.2', 'GIM9.2', 'GIM10.2', 'GIM11.2', 'GIM12.2', 'GIM13.2', 'GIM14.2', 'GIM15.2', 'GIM16.2', 'GIM17.2', 'GIM18.2', 'GIM19.2'], axis=1)

#rescoring the male questionnaire
rev_qs = ['GIM3.1', 'GIM4.1', 'GIM5.1', 'GIM6.1', 'GIM7.1', 'GIM8.1', 'GIM9.1',
          'GIM10.1', 'GIM11.1', 'GIM12.1', 'GIM14.1', 'GIM15.1', 'GIM16.1',
          'GIM17.1', 'GIM18.1', 'GIM19.1', 'GIM3.2', 'GIM4.2', 'GIM5.2', 'GIM6.2',
          'GIM7.2', 'GIM8.2', 'GIM9.2', 'GIM10.2', 'GIM11.2', 'GIM12.2', 'GIM14.2',
          'GIM15.2', 'GIM16.2', 'GIM17.2', 'GIM18.2', 'GIM19.2']

for q in rev_qs:
    m_df[q] = 6 - m_df[q]
m_df.replace(0, np.nan, inplace=True)
m_df.replace(-1, np.nan, inplace=True)

four_qs = ['GIF5.2', 'GIF11.2', 'GIF5.1', 'GIF11.1']
for q in four_qs:
    f_df[q] = f_df[q] - 4.

six_qs = ['GIF6.2', 'GIF12.2', 'GIF6.1', 'GIF12.1']
for q in six_qs:
    f_df[q] = f_df[q] - 6.

f_df.replace({14:1., 11:2., 10:3., 9:4., 8:5., 15: np.nan}, inplace=True)
f_df.replace(6., np.nan, inplace=True)
f_df.replace(-4, np.nan, inplace=True)

#split pre and post via multilevel columns
f_df.columns = f_df.columns.str.split('.', expand=True)
m_df.columns = m_df.columns.str.split('.', expand=True)
f_df = f_df.reorder_levels([1,0], axis=1)
m_df = m_df.reorder_levels([1,0], axis=1)
#final scoring per participant
f_pre_scores = f_df['1'].mean(axis=1)
f_post_scores = f_df['2'].mean(axis=1)

m_pre_scores = m_df['1'].mean(axis=1)
m_post_scores = m_df['2'].mean(axis=1)

m_pre_scores.mean(skipna=True)
m_pre_scores.std(skipna=True)
m_post_scores.mean(skipna=True)
m_post_scores.std(skipna=True)

f_pre_scores.mean(skipna=True)
f_pre_scores.std(skipna=True)
f_post_scores.mean(skipna=True)
f_post_scores.std(skipna=True)

np.mean(m_pre_scores - m_post_scores)

#do recalled gender identity scores vary significantly pre-post?
f_diff = ttest_rel(f_pre_scores, f_post_scores, nan_policy='omit')
f_diff
m_diff = ttest_rel(m_pre_scores, m_post_scores, nan_policy='omit')
m_diff

#how about between genders?
pre_diff = ttest_ind(f_pre_scores, m_pre_scores, nan_policy='omit')
post_diff = ttest_ind(f_post_scores, m_post_scores, nan_policy='omit')

pre_diff
post_diff

f_df['GID Pre'] = f_pre_scores
f_df['GID Post'] = f_post_scores
m_df['GID Pre'] = m_pre_scores
m_df['GID Post'] = m_post_scores

f_df.dropna(axis=0, how='all', inplace=True)
m_df.dropna(axis=0, how='all', inplace=True)

m_df.head()

f_df.to_csv(join(data_dir, 'rescored_gender_identity_female.csv'))
m_df.to_csv(join(data_dir, 'rescored_gender_identity_male.csv'))
