import numpy as np
import pandas as pd
import seaborn as sns
from os.path import join
from nilearn.mass_univariate import permuted_ols
from scipy.stats import pearsonr, spearmanr

data_dir = '/Users/Katie/Dropbox/Projects/physics-retrieval/data'
fig_dir = '/Users/Katie/Dropbox/Projects/physics-retrieval/figures'

df = pd.read_csv(join(data_dir, 'iq+brain+demo.csv'), index_col=0, header=0)
f_gender_df = pd.read_csv(join(data_dir, 'rescored_gender_identity_female-post.csv'), index_col=0, header=0)
m_gender_df = pd.read_csv(join(data_dir, 'rescored_gender_identity_male-post.csv'), index_col=0, header=0)
phy_rt_df = pd.read_csv(join(data_dir, 'retr_physcond_accuracy_by_gender_post.txt'), sep='\t', index_col=1)

df = pd.concat([df, phy_rt_df], axis=1, sort=True)

df_ladies = df[df['Sex'] == 'F']
df_ladies = df_ladies.drop('Sex', axis=1)
df_ladies = pd.concat([df_ladies, f_gender_df], axis=1, sort=False)

df_dudes = df[df['Sex'] == 'M']
df_dudes = df_dudes.drop('Sex', axis=1)
df_dudes = pd.concat([df_dudes, m_gender_df], axis=1, sort=False)

all_gend = pd.concat([f_gender_df, m_gender_df], axis=0, sort=True)
big_df = pd.concat([df, all_gend], axis=1, sort=True)

labels = ['limbic', 'orbitofrontal', 'basal ganglia', 'salience', 'hunger',
          'motor learning', 'frontoparietal', 'hand', 'motor execution', 'higher order visual',
          'lateral visual', 'medial visual', 'default mode', 'cerebellum', 'right central executive',
          'auditory', 'mouth', 'left central executive']

behav = ['Phy48Grade', 'Verbal Comprehension Sum_2','Perceptual Reasoning Sum_2', 'Full Scale IQ_2', 'Total', 'Mean Physics Retrieval Accuracy']

m_gender_df.sort_index(inplace=True)
df_dudes.drop(457, axis=0, inplace=True)
df_dudes.drop(580, axis=0, inplace=True)
df_dudes.drop(335, axis=0, inplace=True)
df_dudes.drop(587, axis=0, inplace=True)

m_phys_corrmats = {}
for i in df_dudes.index:
    corrmat = pd.read_csv(join(data_dir, 'out', '{0}-phy-corrmat.csv'.format(i)), sep=' ', header=None).values
    m_phys_corrmats[i] = np.ravel(corrmat, order='F')

m_phy = np.vstack((m_phys_corrmats.values()))

m_phy_p, m_phy_t,_ = permuted_ols(tested_vars=df_dudes[behav].values, target_vars=m_phy, model_intercept=True, verbose=1, n_perm=10000)

t = m_phy_t[0].reshape((len(labels),len(labels)), order='F')
p = m_phy_p[0].reshape((len(labels),len(labels)), order='F')
m_grade_p = pd.DataFrame(10**-p, index=labels, columns=labels)
m_grade_t = pd.DataFrame(t, index=labels, columns=labels)
m_grade_p.min()

t = m_phy_t[1].reshape((len(labels),len(labels)), order='F')
p = m_phy_p[1].reshape((len(labels),len(labels)), order='F')
m_vc_p = pd.DataFrame(10**-p, index=labels, columns=labels)
m_vc_t = pd.DataFrame(t, index=labels, columns=labels)
m_vc_p.min()

t = m_phy_t[2].reshape((len(labels),len(labels)), order='F')
p = m_phy_p[2].reshape((len(labels),len(labels)), order='F')
m_pr_p = pd.DataFrame(10**-p, index=labels, columns=labels)
m_pr_t = pd.DataFrame(t, index=labels, columns=labels)
m_pr_p.min()

t = m_phy_t[3].reshape((len(labels),len(labels)), order='F')
p = m_phy_p[3].reshape((len(labels),len(labels)), order='F')
m_iq_p = pd.DataFrame(10**-p, index=labels, columns=labels)
m_iq_t = pd.DataFrame(t, index=labels, columns=labels)
m_iq_p.min()

t = m_phy_t[4].reshape((len(labels),len(labels)), order='F')
p = m_phy_p[4].reshape((len(labels),len(labels)), order='F')
m_gender_p = pd.DataFrame(10**-p, index=labels, columns=labels)
m_gender_t = pd.DataFrame(t, index=labels, columns=labels)
m_gender_p.min()

t = m_phy_t[5].reshape((len(labels),len(labels)), order='F')
p = m_phy_p[5].reshape((len(labels),len(labels)), order='F')
m_acc_p = pd.DataFrame(10**-p, index=labels, columns=labels)
m_acc_t = pd.DataFrame(t, index=labels, columns=labels)
m_acc_p.min()


df_ladies.drop(456, axis=0, inplace=True)
df_ladies.drop(590, axis=0, inplace=True)
df_ladies.drop(465, axis=0, inplace=True)
df_ladies.drop(344, axis=0, inplace=True)

f_phys_corrmats = {}
for i in df_ladies.index:
    corrmat = pd.read_csv(join(data_dir, 'out', '{0}-phy-corrmat.csv'.format(i)), sep=' ', header=None).values
    f_phys_corrmats[i] = np.ravel(corrmat, order='F')

f_phy = np.vstack((f_phys_corrmats.values()))

f_phy_p, f_phy_t,_ = permuted_ols(tested_vars=df_ladies[behav].values, target_vars=f_phy, model_intercept=True, verbose=1, n_perm=10000)

t = f_phy_t[0].reshape((len(labels),len(labels)), order='F')
p = f_phy_p[0].reshape((len(labels),len(labels)), order='F')
f_grade_p = pd.DataFrame(10**-p, index=labels, columns=labels)
f_grade_t = pd.DataFrame(t, index=labels, columns=labels)
f_grade_p.min()

t = f_phy_t[1].reshape((len(labels),len(labels)), order='F')
p = f_phy_p[1].reshape((len(labels),len(labels)), order='F')
f_vc_p = pd.DataFrame(10**-p, index=labels, columns=labels)
f_vc_t = pd.DataFrame(t, index=labels, columns=labels)
f_vc_p.min()

t = f_phy_t[2].reshape((len(labels),len(labels)), order='F')
p = f_phy_p[2].reshape((len(labels),len(labels)), order='F')
f_pr_p = pd.DataFrame(10**-p, index=labels, columns=labels)
f_pr_t = pd.DataFrame(t, index=labels, columns=labels)
f_pr_p.min()

t = f_phy_t[3].reshape((len(labels),len(labels)), order='F')
p = f_phy_p[3].reshape((len(labels),len(labels)), order='F')
f_iq_p = pd.DataFrame(10**-p, index=labels, columns=labels)
f_iq_t = pd.DataFrame(t, index=labels, columns=labels)
f_iq_p.min()

t = f_phy_t[4].reshape((len(labels),len(labels)), order='F')
p = f_phy_p[4].reshape((len(labels),len(labels)), order='F')
f_gender_p = pd.DataFrame(10**-p, index=labels, columns=labels)
f_gender_t = pd.DataFrame(t, index=labels, columns=labels)
f_gender_p.min()

t = f_phy_t[5].reshape((len(labels),len(labels)), order='F')
p = f_phy_p[5].reshape((len(labels),len(labels)), order='F')
f_acc_p = pd.DataFrame(10**-p, index=labels, columns=labels)
f_acc_t = pd.DataFrame(t, index=labels, columns=labels)
f_acc_p.min()
