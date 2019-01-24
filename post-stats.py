import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join
from scipy.stats import pearsonr, spearmanr, ttest_rel, ttest_ind, levene, mannwhitneyu
import statsmodels.api as sm

def jili_sidak_mc(data, alpha):
    import math
    import numpy as np

    mc_corrmat = data.corr()
    eigvals, eigvecs = np.linalg.eig(mc_corrmat)

    M_eff = 0
    for eigval in eigvals:
        if abs(eigval) >= 0:
            if abs(eigval) >= 1:
                M_eff += 1
            else:
                M_eff += abs(eigval) - math.floor(abs(eigval))
        else:
            M_eff += 0
    print('Number of effective comparisons: {0}'.format(M_eff))

    #and now applying M_eff to the Sidak procedure
    sidak_p = 1 - (1 - alpha)**(1/M_eff)
    if sidak_p < 0.00001:
        print('Critical value of {:.3f}'.format(alpha),'becomes {:2e} after corrections'.format(sidak_p))
    else:
        print('Critical value of {:.3f}'.format(alpha),'becomes {:.6f} after corrections'.format(sidak_p))
    return sidak_p, M_eff


sns.set_style("whitegrid")

data_dir = '/Users/Katie/Dropbox/Projects/physics-retrieval/data'
df = pd.read_csv(join(data_dir, 'iq+brain+demo.csv'), index_col=0, header=0)
f_gender_df = pd.read_csv(join(data_dir, 'rescored_gender_identity_female.csv'), index_col=0, header=[0,1])
m_gender_df = pd.read_csv(join(data_dir, 'rescored_gender_identity_male.csv'), index_col=0, header=[0,1])
phy_rt_df = pd.read_csv(join(data_dir, 'retr_physcond_accuracy_by_gender_post.txt'), sep='\t', index_col=1)
gen_rt_df = pd.read_csv(join(data_dir, 'retr_gencond_accuracy_by_gender_post.txt'), sep='\t', index_col=1)

df = pd.concat([df, phy_rt_df], axis=1, sort=True)
#we don't care about pre data, only post
f_gender_df.drop(['1'], axis=1, inplace=True)
m_gender_df.drop(['1'], axis=1, inplace=True)


mean = pd.Series(df.mean())
median = pd.Series(df.median())
sdev = pd.Series(df.std())
skew = pd.Series(df.skew())
kurtosis = pd.Series(df.kurtosis())
variance = pd.Series(df.var())
list_of_measures = [mean, median, sdev, variance, skew, kurtosis]

descriptives = pd.concat(list_of_measures, axis=1, sort=False)
descriptives = descriptives.rename({0:'mean', 1:'median', 2:'sdev', 3:'variance', 4:'skew', 5:'kurtosis'}, axis=1)
descriptives.to_csv(join(data_dir, 'decriptives.csv'))

df_ladies = df[df['Sex'] == 'F']
df_ladies = df_ladies.drop('Sex', axis=1)
df_ladies = df_ladies.drop('Session', axis=1)
df_ladies = df_ladies.drop('Gender', axis=1)
df_ladies = df_ladies.drop('Class', axis=1)
df_ladies = df_ladies.drop('Gender And Class', axis=1)
df_ladies = pd.concat([df_ladies, f_gender_df], axis=1, sort=False)

df_dudes = df[df['Sex'] == 'M']
df_dudes = df_dudes.drop('Sex', axis=1)
df_dudes = df_dudes.drop('Session', axis=1)
df_dudes = df_dudes.drop('Gender', axis=1)
df_dudes = df_dudes.drop('Gender And Class', axis=1)
df_dudes = df_dudes.drop('Class', axis=1)
df_dudes = pd.concat([df_dudes, m_gender_df], axis=1, sort=False)

f_gender_df['Masculinity'] = 6 - f_gender_df['2', 'GID Post']
m_gender_df['Masculinity'] = m_gender_df['2', 'GID Post']
df_ladies['Masculinity'] = f_gender_df['Masculinity']
df_dudes['Masculinity'] = m_gender_df['Masculinity']

mean = pd.Series(df_ladies.mean())
median = pd.Series(df_ladies.median())
sdev = pd.Series(df_ladies.std())
skew = pd.Series(df_ladies.skew())
kurtosis = pd.Series(df_ladies.kurtosis())
variance = pd.Series(df_ladies.var())
list_of_measures = [mean, median, sdev, skew, kurtosis]

lady_descriptives = pd.concat(list_of_measures, axis=1, sort=False)
lady_descriptives = lady_descriptives.rename({0:'mean', 1:'median', 2:'sdev', 3:'variance', 4:'skew', 5:'kurtosis'}, axis=1)
lady_descriptives.to_csv(join(data_dir, 'decriptives_f.csv'))

mean = pd.Series(df_dudes.mean())
median = pd.Series(df_dudes.median())
sdev = pd.Series(df_dudes.std())
skew = pd.Series(df_dudes.skew())
kurtosis = pd.Series(df_dudes.kurtosis())
variance = pd.Series(df_dudes.var())
list_of_measures = [mean, median, sdev, skew, kurtosis]

dude_descriptives = pd.concat(list_of_measures, axis=1, sort=False)
dude_descriptives = dude_descriptives.rename({0:'mean', 1:'median', 2:'sdev', 3:'variance', 4:'skew', 5:'kurtosis'}, axis=1)
dude_descriptives.to_csv(join(data_dir, 'decriptives_m.csv'))

#scorr = spearmanr(df)
pcorr = df.corr(method='spearman')


brain = ['fc default mode-left central executive gen',
        'fc default mode-left central executive phy',
        'fc default mode-right central executive gen',
        'fc default mode-right central executive phy',
        'fc hippo-default mode gen', 'fc hippo-default mode phy',
        'fc hippo-left central executive gen',
        'fc hippo-left central executive phy',
        'fc hippo-right central executive gen',
        'fc hippo-right central executive phy',
        'fc left central executive-right central executive gen',
        'fc left central executive-right central executive phy',
        'global efficiency gen', 'global efficiency phy',
        'le default mode gen', 'le default mode phy',
        'le left central executive gen', 'le left central executive phy',
        'le right central executive gen', 'le right central executive phy']
behav = ['Phy48Grade', 'Verbal Comprehension Sum_2','Perceptual Reasoning Sum_2', 'Full Scale IQ_2', 'Total', 'Mean Physics Retrieval Accuracy']

#compare brain measures in physics and general conditions, then separate male & female ppts and repeat
pairs = {}

pairs['dmn_rcen'] = ttest_rel(df['fc default mode-right central executive gen'], df['fc default mode-right central executive phy'])
pairs['dmn_lcen'] = ttest_rel(df['fc default mode-left central executive gen'], df['fc default mode-left central executive phy'])
pairs['hip_dmn'] = ttest_rel(df['fc hippo-default mode gen'], df['fc hippo-default mode phy'])
pairs['hip_lcen'] = ttest_rel(df['fc hippo-left central executive gen'], df['fc hippo-left central executive phy'])
pairs['hip_rcen'] = ttest_rel(df['fc hippo-right central executive gen'], df['fc hippo-right central executive phy'])
pairs['lcen_rcen'] = ttest_rel(df['fc left central executive-right central executive gen'], df['fc left central executive-right central executive phy'])
pairs['gleff'] = ttest_rel(df['global efficiency gen'], df['global efficiency phy'])
pairs['leff_dmn'] = ttest_rel(df['le default mode gen'], df['le default mode phy'])
pairs['leff_lcen'] = ttest_rel(df['le left central executive gen'], df['le left central executive phy'])
pairs['leff_rcen'] = ttest_rel(df['le right central executive gen'], df['le right central executive phy'])

paired_tests = pd.DataFrame.from_dict(pairs, orient='index')
paired_tests = paired_tests.stack()

pairs = {}
pairs['dmn_rcen'] = ttest_rel(df_ladies['fc default mode-right central executive gen'], df_ladies['fc default mode-right central executive phy'])
pairs['dmn_lcen'] = ttest_rel(df_ladies['fc default mode-left central executive gen'], df_ladies['fc default mode-left central executive phy'])
pairs['hip_dmn'] = ttest_rel(df_ladies['fc hippo-default mode gen'], df_ladies['fc hippo-default mode phy'])
pairs['hip_lcen'] = ttest_rel(df_ladies['fc hippo-left central executive gen'], df_ladies['fc hippo-left central executive phy'])
pairs['hip_rcen'] = ttest_rel(df_ladies['fc hippo-right central executive gen'], df_ladies['fc hippo-right central executive phy'])
pairs['lcen_rcen'] = ttest_rel(df_ladies['fc left central executive-right central executive gen'], df_ladies['fc left central executive-right central executive phy'])
pairs['gleff'] = ttest_rel(df_ladies['global efficiency gen'], df_ladies['global efficiency phy'])
pairs['leff_dmn'] = ttest_rel(df_ladies['le default mode gen'], df_ladies['le default mode phy'])
pairs['leff_lcen'] = ttest_rel(df_ladies['le left central executive gen'], df_ladies['le left central executive phy'])
pairs['leff_rcen'] = ttest_rel(df_ladies['le right central executive gen'], df_ladies['le right central executive phy'])

lady_paired_tests = pd.DataFrame.from_dict(pairs, orient='index')
lady_paired_tests = lady_paired_tests.stack()

pairs = {}
pairs['dmn_rcen'] = ttest_rel(df_dudes['fc default mode-right central executive gen'], df_dudes['fc default mode-right central executive phy'])
pairs['dmn_lcen'] = ttest_rel(df_dudes['fc default mode-left central executive gen'], df_dudes['fc default mode-left central executive phy'])
pairs['hip_dmn'] = ttest_rel(df_dudes['fc hippo-default mode gen'], df_dudes['fc hippo-default mode phy'])
pairs['hip_lcen'] = ttest_rel(df_dudes['fc hippo-left central executive gen'], df_dudes['fc hippo-left central executive phy'])
pairs['hip_rcen'] = ttest_rel(df_dudes['fc hippo-right central executive gen'], df_dudes['fc hippo-right central executive phy'])
pairs['lcen_rcen'] = ttest_rel(df_dudes['fc left central executive-right central executive gen'], df_dudes['fc left central executive-right central executive phy'])
pairs['gleff'] = ttest_rel(df_dudes['global efficiency gen'], df_dudes['global efficiency phy'])
pairs['leff_dmn'] = ttest_rel(df_dudes['le default mode gen'], df_dudes['le default mode phy'])
pairs['leff_lcen'] = ttest_rel(df_dudes['le left central executive gen'], df_dudes['le left central executive phy'])
pairs['leff_rcen'] = ttest_rel(df_dudes['le right central executive gen'], df_dudes['le right central executive phy'])

dude_paired_tests = pd.DataFrame.from_dict(pairs, orient='index')
dude_paired_tests = dude_paired_tests.stack()

paired_ttests = pd.DataFrame({'all': paired_tests, 'female': lady_paired_tests, 'male': dude_paired_tests})
paired_ttests = paired_ttests.unstack()

paired_ttests.to_csv(join(data_dir, 'paired-ttests_brain.csv'))

#now, compare male and female ppts on each brain measure
sex_diff = {}
#for key in df_ladies.keys():
#    unequal_var = levene(df_ladies[key], df_dudes[key], center='mean')
#    if unequal_var[1] < 0.05:
#        sex_diff[key] = ttest_ind(df_ladies[key], df_dudes[key], equal_var=True)
#    else:
#        sex_diff[key] = ttest_ind(df_ladies[key], df_dudes[key], equal_var=False)
#should actually be using mannwhitneyu instead of ttest_ind
keys = ['AgeOnScanDate','Phy48Grade','GPA.PreSem',
        'fc default mode-left central executive gen',
                  'fc default mode-left central executive phy',
                 'fc default mode-right central executive gen',
                 'fc default mode-right central executive phy',
                                   'fc hippo-default mode gen',
                                   'fc hippo-default mode phy',
                         'fc hippo-left central executive gen',
                         'fc hippo-left central executive phy',
                        'fc hippo-right central executive gen',
                        'fc hippo-right central executive phy',
       'fc left central executive-right central executive gen',
       'fc left central executive-right central executive phy',
                                       'global efficiency gen',
                                       'global efficiency phy',
                                         'le default mode gen',
                                         'le default mode phy',
                               'le left central executive gen',
                               'le left central executive phy',
                              'le right central executive gen',
                              'le right central executive phy',
                                  'Verbal Comprehension Sum_2',
                                  'Perceptual Reasoning Sum_2',
                                             'Full Scale IQ_2',
                                                       'index',
                             'Mean Physics Retrieval Accuracy',
                                             'Mean Correct RT',
                                           'Mean Incorrect RT',
                                             ('2', 'GID Post'),
                                                 'Masculinity']

for key in keys:
    sex_diff[key] = mannwhitneyu(df_ladies[key], df_dudes[key])

sex_differences = pd.DataFrame.from_dict(sex_diff, orient='index')
sex_differences.to_csv(join(data_dir, 'sex_differences.csv'))

all_gend = pd.concat([f_gender_df, m_gender_df], axis=0, sort=True)
big_df = pd.concat([df, all_gend], axis=1, sort=True)
big_df['Female'] = big_df['Sex']
big_df.replace({'Female': {'M': 1, 'F': 2}}, inplace=True)
big_df['sexXIQ'] = big_df['Full Scale IQ_2'] * big_df['Female']
big_df['sexXgender'] = big_df['Total'] * big_df['Female']
big_df['sexXmasc'] = big_df['Masculinity'] * big_df['Female']
big_df['const'] = 1

behav = ['Phy48Grade', 'Verbal Comprehension Sum_2',
         'Perceptual Reasoning Sum_2', 'Full Scale IQ_2', 'Total', 'Mean Physics Retrieval Accuracy', 'Masculinity']

jili_sidak_mc(big_df[brain],0.05)
jili_sidak_mc(df_dudes[behav],0.05)
jili_sidak_mc(df_ladies[behav],0.05)

scorrs = {}
for key in brain:
    for meas in behav:
        scorrs[key, meas] = spearmanr(big_df[key], big_df[meas], nan_policy='omit')
all_corr = pd.DataFrame.from_dict(scorrs, orient='index')
all_corr.to_csv(join(data_dir, 'corr_all_brain_meas.csv'))

mcorrs = {}
for key in brain:
    for meas in behav:
        mcorrs[key, meas] = spearmanr(df_dudes[key], df_dudes[meas], nan_policy='omit')

male_corr = pd.DataFrame.from_dict(mcorrs, orient='index')
male_corr.min()
male_corr.to_csv(join(data_dir, 'corr_male_brain_meas.csv'))

fcorrs = {}
for key in brain:
    for meas in behav:
        fcorrs[key, meas] = spearmanr(df_ladies[key], df_ladies[meas], nan_policy='omit')

female_corr = pd.DataFrame.from_dict(fcorrs, orient='index')
female_corr.min()
female_corr.to_csv(join(data_dir, 'corr_female_brain_meas.csv'))

spearmanr(df_ladies['Total'].dropna(), df_ladies['Mean Physics Retrieval Accuracy'].dropna())
spearmanr(df_dudes['Total'].dropna(), df_dudes['Mean Physics Retrieval Accuracy'].dropna())
spearmanr(df_ladies['Total'].dropna(), df_ladies['Mean Correct RT'].dropna())
spearmanr(df_dudes['Total'], df_dudes['Mean Correct RT'], nan_policy='omit')
spearmanr(df_ladies['Total'].dropna(), df_ladies['Mean Incorrect RT'].dropna())
spearmanr(df_dudes['Total'], df_dudes['Mean Incorrect RT'], nan_policy='omit')

f_gid_behav = {}
m_gid_behav = {}
gid_behav = {}
for key in behav:
    f_gid_behav[key] = spearmanr(df_ladies['Total'], df_ladies[key], nan_policy='omit')
    m_gid_behav[key] = spearmanr(df_dudes['Total'], df_dudes[key], nan_policy='omit')
    gid_behav[key] = spearmanr(big_df['Total'], big_df[key], nan_policy='omit')

f_gid_behav_corr = pd.DataFrame.from_dict(f_gid_behav, orient='index')
f_gid_behav_corr.to_csv(join(data_dir, 'corr_female_gid_meas.csv'))
m_gid_behav_corr = pd.DataFrame.from_dict(m_gid_behav, orient='index')
m_gid_behav_corr.to_csv(join(data_dir, 'corr_male_gid_meas.csv'))
gid_behav_corr = pd.DataFrame.from_dict(gid_behav, orient='index')
gid_behav_corr.to_csv(join(data_dir, 'corr_gid_meas.csv'))

########Now for some regressions!########

conn = ['fc default mode-left central executive gen',
       'fc default mode-left central executive phy',
       'fc default mode-right central executive gen',
       'fc default mode-right central executive phy',
       'fc hippo-default mode gen', 'fc hippo-default mode phy',
       'fc hippo-left central executive gen',
       'fc hippo-left central executive phy',
       'fc hippo-right central executive gen',
       'fc hippo-right central executive phy',
       'fc left central executive-right central executive gen',
       'fc left central executive-right central executive phy']
eff = ['global efficiency gen', 'global efficiency phy', 'le default mode gen',
       'le default mode phy', 'le left central executive gen',
       'le left central executive phy', 'le right central executive gen',
       'le right central executive phy']

demo_iq = ['Female', 'AgeOnScanDate', 'Full Scale IQ_2']
academic = ['Female', 'Phy48Grade', 'SexXGrade']
sex_gend = ['Female', 'Masculinity', 'sexXmasc']
non_brain = ['Female', 'Verbal Comprehension Sum_2', 'Perceptual Reasoning Sum_2',
             'Full Scale IQ_2', 'sexXIQ', 'Phy48Grade', 'SexXGrade', 'AgeOnScanDate']

#does IQ predict brain connectivity, controlling for age and sex?
iq_summary = pd.DataFrame(columns=['BIC', 'F', 'p(F)', 'df_model', 'R^2'], index=brain)
iq_pvals = {}
iq_params = {}

for brain_var in brain:
    #print('******************************{0}********************************'.format(brain_var))
    y = big_df.loc[:, brain_var].dropna()
    X = big_df.loc[:, demo_iq].dropna()
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    #print(model.summary())
    iq_summary.at[brain_var,'BIC'] = model.bic
    iq_summary.at[brain_var,'F'] = model.fvalue
    iq_summary.at[brain_var,'df_model'] = model.df_model
    iq_summary.at[brain_var,'p(F)'] = model.f_pvalue
    iq_summary.at[brain_var,'R^2'] = model.rsquared

    iq_params[brain_var] = model.params
    iq_pvals[brain_var] = model.pvalues
iq_summary.min()

#what about sex and gender?
sg_summary = pd.DataFrame(columns=['BIC', 'F', 'p(F)', 'df_model', 'R^2'], index=brain)
sg_pvals = {}
sg_params = {}

no_missing_gender = big_df
no_missing_gender.drop(335, axis=0, inplace=True)
no_missing_gender.drop(344, axis=0, inplace=True)
no_missing_gender.drop(465, axis=0, inplace=True)
no_missing_gender.drop(587, axis=0, inplace=True)

for brain_var in brain:
    #print('******************************{0}********************************'.format(brain_var))
    y = no_missing_gender.loc[:, brain_var].dropna()
    X = no_missing_gender.loc[:, sex_gend].dropna()
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    #res = model.resid # residuals
    #fig = sm.qqplot(res)
    #plt.show()
    #print(model.summary())
    sg_summary.at[brain_var,'BIC'] = model.bic
    sg_summary.at[brain_var,'F'] = model.fvalue
    sg_summary.at[brain_var,'df_model'] = model.df_model
    sg_summary.at[brain_var,'p(F)'] = model.f_pvalue
    sg_summary.at[brain_var,'R^2'] = model.rsquared

    sg_params[brain_var] = model.params
    sg_pvals[brain_var] = model.pvalues
sg_pvals

#how do brain + iq + gender predict academic performance
ac_summary = pd.DataFrame(columns=['BIC', 'F', 'p(F)', 'df_model', 'R^2'], index=brain)
ac_pvals = {}
ac_params = {}


for brain_var in brain:
    vars = list(set(demo_iq)) + [brain_var]
    #vars += [brain_var]
    #print('******************************{0}********************************'.format(brain_var))
    y = big_df.loc[:, 'Phy48Grade'].dropna()
    X = big_df.loc[:, vars].dropna()
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    #print(model.summary())
    ac_summary.at[brain_var,'BIC'] = model.bic
    ac_summary.at[brain_var,'F'] = model.fvalue
    ac_summary.at[brain_var,'df_model'] = model.df_model
    ac_summary.at[brain_var,'p(F)'] = model.f_pvalue
    ac_summary.at[brain_var,'R^2'] = model.rsquared

    ac_params[brain_var] = model.params
    ac_pvals[brain_var] = model.pvalues
ac_summary.min()
