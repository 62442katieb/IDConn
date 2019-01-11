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
from scipy.stats import ttest_ind


#less important for plotting

import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from statsmodels.sandbox.stats.multicomp import multipletests
from nilearn.mass_univariate import permuted_ols

data_dir = '/Users/Katie/Dropbox/Projects/physics-retrieval/data'
fig_dir = '/Users/Katie/Dropbox/Projects/physics-retrieval/figures'

df = pd.read_excel(join(data_dir, 'gender-identity-prepost.xlsx'), index_col=0, header=0)

df.head()
df.keys()
m_df = df[df['Sex.1'] == 1]
m_df = m_df.drop(['Sex.1', 'GIM1.1', 'GIM2.1', 'GIM3.1', 'GIM4.1', 'GIM5.1', 'GIM6.1', 'GIM7.1', 'GIM8.1', 'GIM9.1', 'GIM10.1', 'GIM11.1', 'GIM12.1', 'GIM13.1', 'GIM14.1', 'GIM15.1', 'GIM16.1', 'GIM17.1', 'GIM18.1', 'GIM19.1', 'GIF1.1', 'GIF2.1', 'GIF3.1', 'GIF4.1', 'GIF5.1', 'GIF6.1', 'GIF7.1', 'GIF8.1', 'GIF9.1', 'GIF10.1', 'GIF11.1', 'GIF12.1', 'GIF13.1', 'GIF14.1', 'GIF15.1', 'GIF16.1', 'GIF17.1', 'GIF18.1', 'GIF19.1', 'GIF1.2', 'GIF2.2', 'GIF3.2', 'GIF4.2', 'GIF5.2', 'GIF6.2', 'GIF7.2', 'GIF8.2', 'GIF9.2', 'GIF10.2', 'GIF11.2', 'GIF12.2','GIF13.2', 'GIF14.2', 'GIF15.2', 'GIF16.2', 'GIF17.2', 'GIF18.2','GIF19.2', 'GIM13.2'], axis=1)
f_df = df[df['Sex.1'] == 2]
f_df = f_df.drop(['Sex.1', 'GIM1.1', 'GIM2.1', 'GIM3.1', 'GIM4.1', 'GIM5.1', 'GIM6.1', 'GIM7.1', 'GIM8.1', 'GIM9.1', 'GIM10.1', 'GIM11.1', 'GIM12.1', 'GIM13.1', 'GIM14.1', 'GIM15.1', 'GIM16.1', 'GIM17.1', 'GIM18.1',  'GIM19.1', 'GIF1.1', 'GIF2.1', 'GIF3.1', 'GIF4.1', 'GIF5.1', 'GIF6.1', 'GIF7.1', 'GIF8.1', 'GIF9.1', 'GIF10.1', 'GIF11.1', 'GIF12.1', 'GIF13.1', 'GIF14.1', 'GIF15.1', 'GIF16.1', 'GIF17.1', 'GIF18.1', 'GIF19.1', 'GIM1.2', 'GIM2.2', 'GIM3.2', 'GIM4.2', 'GIM5.2', 'GIM6.2', 'GIM7.2', 'GIM8.2', 'GIM9.2', 'GIM10.2', 'GIM11.2', 'GIM12.2', 'GIM13.2', 'GIM14.2', 'GIM15.2', 'GIM16.2', 'GIM17.2', 'GIM18.2', 'GIM19.2', 'GIF13.2'], axis=1)
m_df.index
f_df.index
#rescoring the male questionnaire
rev_qs = ['GIM3.2', 'GIM4.2', 'GIM5.2', 'GIM6.2', 'GIM7.2', 'GIM8.2', 'GIM9.2', 'GIM10.2', 'GIM11.2', 'GIM12.2', 'GIM14.2', 'GIM15.2', 'GIM16.2', 'GIM17.2', 'GIM18.2', 'GIM19.2']

for q in rev_qs:
    m_df[q] = 6 - m_df[q]
m_df.replace(0, 'NA', inplace=True)
m_df.replace(-1, 'NA', inplace=True)

four_qs = ['GIF5.2', 'GIF11.2']
for q in four_qs:
    f_df[q] = f_df[q] - 4.

six_qs = ['GIF6.2', 'GIF12.2']
for q in six_qs:
    f_df[q] = f_df[q] - 6.

f_df.replace({14:1., 11:2., 10:3., 9:4., 8:5., 15: np.nan}, inplace=True)
f_df.replace(6., np.nan, inplace=True)
f_df.replace(-4, np.nan, inplace=True)

#final scoring per participant
m_scores = m_df.mean(axis=1)
f_scores = f_df.mean(axis=1)

m_scores.std()
m_scores.mean()
f_scores.std()
f_scores.mean()

f_df['Total'] = f_scores
m_df['Total'] = m_scores

f_df.dropna(axis=0, how='all', inplace=True)
m_df.dropna(axis=0, how='all', inplace=True)

f_df.to_csv(join(data_dir, 'rescored_gender_identity_female-post.csv'))
m_df.to_csv(join(data_dir, 'rescored_gender_identity_male-post.csv'))
