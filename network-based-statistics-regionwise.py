import numpy as np
import pandas as pd
import bct
from os.path import join

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t

data_dir = '/Users/Katie/Dropbox/Projects/physics-retrieval/data'
df = pd.read_csv(join(data_dir, 'iq+brain+demo.csv'), index_col=0, header=0)

labels = ['limbic', 'limbic', 'orbitofrontal', 'orbitofrontal', 'basal ganglia',
          'salience', 'salience', 'salience', 'hunger', 'hunger', 'hunger',
          'hunger', 'hunger', 'hunger', 'hunger', 'motor learning', 'frontoparietal',
          'frontoparietal', 'frontoparietal', 'hand', 'hand', 'hand', 'motor execution',
          'motor execution', 'higher order visual', 'higher order visual',
          'lateral visual', 'lateral visual', 'medial visual', 'default mode',
          'default mode', 'default mode', 'default mode', 'default mode', ' cerebellum',
          'right central executive', 'right central executive', 'right central executive',
          'right central executive', 'right central executive', 'auditory', 'auditory',
          'mouth', 'mouth', 'left central executive', 'left central executive',
          'left central executive']

#temporary, until I decide how to deal with subjects 321 and 618
#who have different numbers of regions for the two runs
df.drop(321, axis=0, inplace=True)
df.drop(618, axis=0, inplace=True)

m_df = df[df['Sex'] == 'M']
m_df.to_csv(join(data_dir, 'male_df.csv'))
f_df = df[df['Sex'] == 'F']
f_df.to_csv(join(data_dir, 'female_df.csv'))

pd.read_csv(join(data_dir, 'out', '102-phy-corrmat-regionwise.csv'), header=0, index_col=0).values

m_phys_corrmats = []
for i in m_df.index:
    corrmat = pd.read_csv(join(data_dir, 'out', '{0}-phy-corrmat-regionwise.csv'.format(i)), header=0, index_col=0)
    m_phys_corrmats.append(corrmat.values)

m_phy = np.dstack((m_phys_corrmats))

f_phys_corrmats = []
for i in f_df.index:
    corrmat = pd.read_csv(join(data_dir, 'out', '{0}-phy-corrmat-regionwise.csv'.format(i)), header=0, index_col=0)
    f_phys_corrmats.append(corrmat.values)

f_phy = np.dstack((f_phys_corrmats))

#run nbs for female > male in physics
d_freedom = len(f_df.index) - 2
d_freedom
t_crit_sex = t.ppf(0.95,d_freedom)

m_phy.shape

phy_pval, phy_adj, phy_null = bct.nbs_bct(f_phy, m_phy, thresh=t_crit_sex, k=1000, tail='left', paired=False, verbose=False)
pd.DataFrame(phy_adj, index=labels, columns=labels).to_csv(join(data_dir, 'm-gt-f_phy-regionwise_comp_adj_{0}.csv'.format(phy_pval)))

phy_pval, phy_adj, phy_null = bct.nbs_bct(f_phy, m_phy, thresh=t_crit_sex, k=1000, tail='right', paired=False, verbose=False)
pd.DataFrame(phy_adj, index=labels, columns=labels).to_csv(join(data_dir, 'f-gt-m_phy-regionwise_comp_adj_{0}.csv'.format(phy_pval)))

m_gen_corrmats = []
for i in m_df.index:
    corrmat = pd.read_csv(join(data_dir, 'out', '{0}-gen-corrmat-regionwise.csv'.format(i)), header=0, index_col=0)
    m_gen_corrmats.append(corrmat.values)

m_gen = np.dstack((m_gen_corrmats))

f_gen_corrmats = []
for i in f_df.index:
    f_gen_corrmats.append(pd.read_csv(join(data_dir, 'out', '{0}-gen-corrmat-regionwise.csv'.format(i)), header=0, index_col=0).values)

f_gen = np.dstack((f_gen_corrmats))


all_gen = np.dstack((m_gen,f_gen))
all_phy = np.dstack((m_phy,f_phy))

#run nbs for whole sample phy > gen
d_freedom = len(df.index) - 2
d_freedom
t_crit_all = t.ppf(0.95,d_freedom)

gen_gt_phy_pval, gen_gt_phy_adj, null = bct.nbs_bct(all_gen, all_phy, thresh=t_crit_all, k=1000, tail='right', paired=True, verbose=False)
pd.DataFrame(gen_gt_phy_adj, index=labels, columns=labels).to_csv(join(data_dir, 'gen-gt-phy_gen-regionwise_comp_adj_{0}.csv'.format(gen_gt_phy_pval)))
phy_gt_gen_pval, phy_gt_gen_adj, null = bct.nbs_bct(all_gen, all_phy, thresh=t_crit_all, k=1000, tail='left', paired=True, verbose=False)
pd.DataFrame(phy_gt_gen_adj, index=labels, columns=labels).to_csv(join(data_dir, 'phy-gt-gen-regionwise_comp_adj_{0}.csv'.format(phy_gt_gen_pval)))


#run nbs for female > male in physics
d_freedom = len(f_df.index) - 1
d_freedom
t_crit_f = t.ppf(0.95,d_freedom)
gen_nbs_pval, gen_nbs_adj, gen_nbs_null = bct.nbs_bct(f_gen, m_gen, thresh=t_crit_f, k=1000, tail='left', paired=False, verbose=False)
pd.DataFrame(gen_nbs_adj, index=labels, columns=labels).to_csv(join(data_dir, 'm-gt-f_gen-regionwise_comp_adj_{0}.csv'.format(gen_nbs_pval)))

gen_nbs_pval, gen_nbs_adj, gen_nbs_null = bct.nbs_bct(f_gen, m_gen, thresh=t_crit_f, k=1000, tail='right', paired=False, verbose=False)
pd.DataFrame(gen_nbs_adj, index=labels, columns=labels).to_csv(join(data_dir, 'f-gt-m_gen-regionwise_comp_adj_{0}.csv'.format(gen_nbs_pval)))

f_pval, f_adj, f_null = bct.nbs_bct(f_gen, f_phy, thresh=t_crit_f, k=1000, tail='right', paired=False, verbose=False)
pd.DataFrame(f_adj, index=labels, columns=labels).to_csv(join(data_dir, 'f_gen-gt-phy-regionwise_comp_adj_{0}.csv'.format(f_pval)))

d_freedom = len(m_df.index) - 1
d_freedom
t_crit_m = t.ppf(0.95,d_freedom)

m_pval, m_adj, m_null = bct.nbs_bct(m_gen, m_phy, thresh=t_crit_m, k=1000, tail='right', paired=False, verbose=False)
pd.DataFrame(m_adj, index=labels, columns=labels).to_csv(join(data_dir, 'm_gen-gt-phy-regionwise_comp_adj_{0}.csv'.format(m_pval)))

np.sum(m_adj - gen_gt_phy_adj)
