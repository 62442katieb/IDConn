import numpy as np
import pandas as pd
import seaborn as sns
from os import makedirs
from os.path import join, exists
from nilearn.plotting import plot_connectome, plot_roi, find_parcellation_cut_coords
import bct
from datetime import datetime
from nilearn.mass_univariate import permuted_ols
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import KNNImputer

def corrmat_to_samples_by_features(subjects, session, task, condition, mask, tau, order='F', verbose=False):
    # read in every person's connectivity matrix (yikes)
    # flatten into features (edges) per sample (subject)
    # one task & condition at a time, I think. otherwise it becomes a memory issue
    conn_df = pd.DataFrame(index=subjects, columns=np.arange(0, 268**2))

    for subject in subjects:
        corrmat_path = join(sink_dir, 'corrmats',
                            '{0}-session-{1}_{2}-{3}_{4}-corrmat.csv'.format(subject, session, task, condition, mask))
        if verbose:
            print('sub-{0}'.format(subject))
            print('corrmat at {0}'.format())
        try:
            corrmat = np.genfromtxt(corrmat_path, delimiter=' ')
            thresh_corrmat = bct.threshold_proportional(corrmat, tau, copy=True)
            conn_df.at[subject] = np.ravel(corrmat, order='F')
        except Exception as e:
            if verbose:
                print(subject, e)
    return conn_df

sns.set(context='poster', style='ticks')

#color palettes for plotting significant results
crayons_l = sns.crayon_palette(['Vivid Tangerine', 'Cornflower'])
crayons_d = sns.crayon_palette(['Brick Red', 'Midnight Blue'])
grays = sns.light_palette('#999999', n_colors=3, reverse=True)
f_2 = sns.crayon_palette(['Red Orange', 'Vivid Tangerine'])
m_2 = sns.crayon_palette(['Cornflower', 'Cerulean'])

#list of all subjects for reading in correlation matrices
subjects = ['101', '102', '103', '104', '106', '107', '108', '110', '212', '213',
            '214', '215', '216', '217', '218', '219', '320', '321', '322', '323',
            '324', '325', '327', '328', '329', '330', '331', '332', '333', '334',
            '335', '336', '337', '338', '339', '340', '341', '342', '343', '344',
            '345', '346', '347', '348', '349', '350', '451', '452', '453', '455',
            '456', '457', '458', '459', '460', '462', '463', '464', '465', '467',
            '468', '469', '470', '502', '503', '571', '572', '573', '574', '575',
            '577', '578', '579', '580', '581', '582', '584', '585', '586', '587',
            '588', '589', '590', '591', '592', '593', '594', '595', '596', '597',
            '598', '604', '605', '606', '607', '608', '609', '610', '611', '612',
            '613', '614', '615', '616', '617', '618', '619', '620', '621', '622',
            '623', '624', '625', '626', '627', '628', '629', '630', '631', '633',
            '634']
#subjects = ['101', '102']

#input and output directories
data_dir = '/Users/kbottenh/Dropbox/Projects/physics-retrieval/data'
sink_dir = '/Users/kbottenh/Dropbox/Projects/physics-retrieval/data/output'
fig_dir = '/Users/kbottenh/Dropbox/Projects/physics-retrieval/figures/'

#all analyses are repeated with two different brain parcellations 
#want to ensure that redsults are in the brain, not in the parcellation
masks = {'shen2015': '/Users/kbottenh/Dropbox/Projects/physics-retrieval/shen2015_2mm_268_parcellation.nii.gz', 
         'craddock2012': '/Users/kbottenh/Dropbox/Projects/physics-retrieval/craddock2012_tcorr05_2level_270_2mm.nii.gz'}

#results of previous analyses show that these WAIS scores are related
#to accuracy during these tasks
tasks = {'fci': {'iqs': ['PRI2', 'FSIQ2', 'deltaPRI', 'deltaFSIQ']},
         'retr': {'iqs': ['VCI2', 'WMI2']}
         }

tau = 0.31

#read in dataset
big_df = pd.read_csv(join(data_dir, 
                          'rescored', 
                          'physics_learning-local_efficiency-BayesianImpute.csv'), 
                     index_col=0, header=0)

#calculate derived IQ measures
iqs = ['VCI', 'WMI', 'PSI', 'PRI', 'FSIQ']
for iq in iqs:
    big_df['delta{0}'.format(iq)] = big_df['{0}2'.format(iq)] - big_df['{0}1'.format(iq)]
    big_df['delta{0}XSex'.format(iq)] = big_df['delta{0}'.format(iq)] * big_df['F']
    big_df['{0}2XSex'.format(iq)] = big_df['{0}2'.format(iq)] * big_df['F']
    big_df['delta{0}XClass'.format(iq)] = big_df['delta{0}'.format(iq)] * big_df['Mod']
    big_df['{0}2XClass'.format(iq)] = big_df['{0}2'.format(iq)] * big_df['Mod']
    big_df['SexXClass'] = big_df['F'] * big_df['Mod']
    big_df['delta{0}XSexXClass'.format(iq)] = big_df['delta{0}'.format(iq)] * big_df['SexXClass']
    big_df['{0}2XSexXClass'.format(iq)] = big_df['{0}2'.format(iq)] * big_df['SexXClass']

#set the level of Type I error you're comfortable with
#alpha is the probability of a false positive
alpha = 0.1

#now correct alpha for multiple comparisons
n_tests = 2 + 4 #automate this from # tasks + # DVs of interest
#Sidak correction
adj_a = 1 - (1 - alpha)**(1/n_tests)
nloga = -np.log10(adj_a)

#setting up a dataframe for storing the max nlogp value per parameter per regression
#running regressions for each mask, for each task, for each significantly associated IQ
variables = ['iq', 'iqXSex', 'iqXClass', 'iqXSexXClass', 'SexXClass', 'F', 'Mod', 'Age', 'StrtLvl', 'fd']
index = pd.MultiIndex.from_product([masks.keys(), tasks.keys(), variables])
sig = pd.DataFrame(index=index)

#running each permuted OLS regression this many times
n_perm = 10000
#creating figures automatically for regressions with significant edges
node_size = 10

#run all regressions for all task-IQ combos once for each parcellation
for mask in masks.keys():
    #for making connectome figures, read in the relevant parcellation
    mask_nii = masks[mask]
    #and extract coordinates per node/region
    coords = find_parcellation_cut_coords(labels_img=mask_nii)

    #run regressions per task, only reading in all subjects' corrmats 
    #done separately for each task & condition and removing in between bc memory
    for task in tasks.keys():
        #only testing IQs associated with accuracy on this task
        iqs = tasks[task]['iqs']

        #read in all subjects' correlation matrices and flatten into feature vectors
        conn_df = corrmat_to_samples_by_features(subjects, 1, task, 'Physics', mask, tau)
        conn_df.index = conn_df.index.astype(int)
        conns = list(set(conn_df.columns))

        #smush connectivity features and other variables into one big dataframe
        all_data = pd.concat([big_df, conn_df], axis=1)

        #impute missing values using KNN which is robust to large amounts of missingness
        #but not permuted, which is a bummer...
        #although this dataset is a little big for a permuted approach
        brain_impute = KNNImputer(n_neighbors=100, weights='distance')
        imp_mat = brain_impute.fit_transform(all_data.drop(['Sex', 'Class.Type'], axis=1))
        imp_df = pd.DataFrame(data=imp_mat,
                              columns=all_data.drop(['Sex', 'Class.Type'], axis=1).columns,
                              index=all_data.index)
        #imp_df = imp_df.astype('float')

        
        for iq in iqs:
            #fill in IV list with the WAIS measure for this regression
            reg_vars = []
            for var in variables:
                if 'iq' in var:
                    reg_vars.append(var.replace('iq', iq))
                elif 'fd' in var:
                    reg_vars.append(var.replace('fd', 'post phys {0} fd'.format(task)))
                else:
                    reg_vars.append(var)
            #run a separate permuted OLS for each covariate
            #this is the only way to get significant edges associated with each covariate
            #only one covariate "of interest" allowed per permuted_ols() call 
            for i in range(len(reg_vars)):
                var_of_interest = reg_vars[i]
                covariates = list(set(reg_vars) - set([var_of_interest]))
                print(var_of_interest, variables[i])
                print('post phys {0} {1} conns ~ {2} + {3}'.format(task, mask, var_of_interest, covariates))
                print(datetime.now())
                #permuted_ols returns a matrix of -log(p) values, tvalues, and dist
                p, t, _ = permuted_ols(imp_df[var_of_interest],
                                       imp_df[conns],
                                       imp_df[covariates],
                                       n_perm=n_perm, n_jobs=8, verbose=1)
                #save max -log(p) value to see which covairate in which regressions were significant
                sig.at[(mask, task, variables[i]), iq] = np.max(p)
                sig.to_csv(join(sink_dir, 'permuted_ols-conn~iq-maxnlogp-a={0}.csv'.format(alpha)))
                #save out 
                if np.max(p) > nloga:
                    tmap = np.reshape(t, (268, 268), order='F')
                    pmap = np.reshape(p, (268, 268), order='F')

                    tdf = pd.DataFrame(tmap, columns=np.arange(
                        1, 269), index=np.arange(1, 269))
                    tdf.fillna(0, inplace=True)
                    tdf.to_csv(join(sink_dir, '{0}-{1}_phys-{2}_{3}-tvals_a={4}.csv'.format(mask, task, iq, variables[i], alpha)))
                    pdf = pd.DataFrame(pmap, columns=np.arange(
                        1, 269), index=np.arange(1, 269))
                    pdf.fillna(0, inplace=True)
                    pdf.to_csv(join(sink_dir, '{0}-{1}_phys-{2}_{3}-pvals_a={4}.csv'.format(mask, task, iq, variables[i], alpha)))
                    sig_edges = tdf[pdf >= nloga]
                    sig_edges.to_csv(
                        join(sink_dir, '{0}-{1}_phys-{2}_{3}-sig_edges_a={4}.csv'.format(mask, task, iq, variables[i], alpha)))

                    q = plot_connectome(sig_edges, coords, node_size=node_size)
                    q.savefig(
                        join(fig_dir, '{0}-{1}_phys-{2}_{3}-sig_edges_a={4}.png'.format(mask, task, iq, variables[i], alpha)), dpi=300)
