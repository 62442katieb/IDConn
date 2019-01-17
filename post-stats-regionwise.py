import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join
from nilearn.mass_univariate import permuted_ols
import bct
from scipy.stats import pearsonr, spearmanr

sns.set_style("whitegrid")

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
          'mouth', 'mouth', 'right central executive', 'left central executive',
          'left central executive']
#remove all network-wise brain measures
df.drop(['fc default mode-left central executive gen','fc default mode-left central executive phy', 'fc default mode-right central executive gen', 'fc default mode-right central executive phy', 'fc hippo-default mode gen', 'fc hippo-default mode phy', 'fc hippo-left central executive gen', 'fc hippo-left central executive phy', 'fc hippo-right central executive gen', 'fc hippo-right central executive phy', 'fc left central executive-right central executive gen', 'fc left central executive-right central executive phy', 'global efficiency gen', 'global efficiency phy', 'le default mode gen', 'le default mode phy', 'le left central executive gen', 'le left central executive phy', 'le right central executive gen', 'le right central executive phy'], axis=1, inplace=True)
df.keys()

behav = ['Phy48Grade', 'GPA.PreSem', 'Verbal Comprehension Sum_2', 'Perceptual Reasoning Sum_2', 'Full Scale IQ_2']

#temporary, until I decide how to deal with subjects 321 and 618
#who have different numbers of regions for the two runs
df.drop(321, axis=0, inplace=True)
df.drop(618, axis=0, inplace=True)


#####################################################################
########### testing graph theoretic measures (eff, leff) ############


#geffs = pd.DataFrame(index=df.index, columns=np.arange(0,1,0.1))
for i in df.index:
    corrmat = pd.read_csv(join(data_dir, 'out', '{0}-phy-corrmat-regionwise.csv'.format(i)), header=0, index_col=0)
    gw = []
    for p in np.arange(0,1,0.1):
        corrmat_thresh = bct.threshold_proportional(corrmat.values, p)
        gw.append(bct.efficiency_wei(corrmat_thresh))
    #geffs.at[i] = gw
    df.at[i,'Global Efficiency Physics'] = np.trapz(gw[4:9], dx=0.1)
    corrmat = pd.read_csv(join(data_dir, 'out', '{0}-gen-corrmat-regionwise.csv'.format(i)), header=0, index_col=0)
    gw=[]
    for p in np.arange(0,1,0.1):
        corrmat_thresh = bct.threshold_proportional(corrmat.values, p)
        gw.append(bct.efficiency_wei(corrmat_thresh))
    #geffs.at[i] = gw
    df.at[i,'Global Efficiency General'] = np.trapz(gw[4:9], dx=0.1)

f, ax = plt.subplots()
sns.lineplot(x=np.arange(0,1,0.1), y=geffs.mean(axis=0))
sns.lineplot(x=np.arange(0,1,0.1), y=geffs.std(axis=0))

m_df = df[df['Sex'] == 'M']
f_df.index.shape
f_df = df[df['Sex'] == 'F']
#and now we test whether global efficiency is related to any of our behavioral measures
for meas in behav:
    print('\n\nAll Students:\n')
    print('Global Efficiency Physics, {0}: {1}'.format(meas, spearmanr(df['Global Efficiency Physics'], df[meas])))
    print('Global Efficiency General, {0}: {1}'.format(meas, spearmanr(df['Global Efficiency General'], df[meas])))
    print('\nMale Students:\n')
    print('Global Efficiency Physics, {0}: {1}'.format(meas, spearmanr(m_df['Global Efficiency Physics'], m_df[meas])))
    print('Global Efficiency General, {0}: {1}'.format(meas, spearmanr(m_df['Global Efficiency General'], m_df[meas])))
    print('\nFemale Students:\n')
    print('Global Efficiency Physics, {0}: {1}'.format(meas, spearmanr(f_df['Global Efficiency Physics'], f_df[meas])))
    print('Global Efficiency General, {0}: {1}'.format(meas, spearmanr(f_df['Global Efficiency General'], f_df[meas])))


#####################################################################
######## connection-wise testing for relationships with behav########

m_phys_corrmats = []
for i in m_df.index:
    corrmat = pd.read_csv(join(data_dir, 'out', '{0}-phy-corrmat-regionwise.csv'.format(i)), header=0, index_col=0)
    corrmat = np.ravel(corrmat.values, order='F')
    m_phys_corrmats.append(corrmat)

m_phy = np.vstack((m_phys_corrmats))

m_phy_ols_p, m_phy_ols_t, _ = permuted_ols(m_df[behav].values, m_phy)

f_phys_corrmats = []
for i in f_df.index:
    corrmat = pd.read_csv(join(data_dir, 'out', '{0}-phy-corrmat-regionwise.csv'.format(i)), header=0, index_col=0)
    corrmat = np.ravel(corrmat.values, order='F')
    f_phys_corrmats.append(corrmat)

f_phy = np.vstack((f_phys_corrmats))

f_phy_ols_p, f_phy_ols_t, _ = permuted_ols(f_df[behav].values, f_phy)


m_gen_corrmats = []
for i in m_df.index:
    corrmat = pd.read_csv(join(data_dir, 'out', '{0}-gen-corrmat-regionwise.csv'.format(i)), header=0, index_col=0)
    corrmat = np.ravel(corrmat.values, order='F')
    m_gen_corrmats.append(corrmat)

m_gen = np.vstack((m_gen_corrmats))

m_gen_ols_p, m_gen_ols_t, _ = permuted_ols(m_df[behav].values, m_gen)


f_gen_corrmats = []
for i in f_df.index:
    corrmat = pd.read_csv(join(data_dir, 'out', '{0}-gen-corrmat-regionwise.csv'.format(i)), header=0, index_col=0)
    corrmat = np.ravel(corrmat.values, order='F')
    f_gen_corrmats.append(corrmat)

f_gen = np.vstack((f_gen_corrmats))
f_gen_ols_p, f_gen_ols_t, _ = permuted_ols(f_df[behav].values, f_gen)


all_gen = np.dstack((m_gen,f_gen))
all_phy = np.dstack((m_phy,f_phy))
