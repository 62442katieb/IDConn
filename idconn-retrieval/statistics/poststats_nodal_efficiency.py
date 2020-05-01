import numpy as np
import pandas as pd
import seaborn as sns
import nibabel as nib
import bct
from os import makedirs
from matplotlib.colors import LinearSegmentedColormap
from os.path import join, exists
from nilearn.plotting import plot_glass_brain, plot_roi, find_parcellation_cut_coords
#import bct
import datetime
from nilearn.mass_univariate import permuted_ols
from scipy.stats import pearsonr, spearmanr
sns.set_context('poster', font_scale=0.85)
import matplotlib.pyplot as plt

def jili_sidak_mc(data, alpha):
    import math
    import numpy as np

    mc_corrmat = data.corr()
    mc_corrmat.fillna(0, inplace=True)
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


# In[3]:


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

sink_dir = '/Users/kbottenh/Dropbox/Projects/physics-retrieval/data/output'
data_dir = '/Users/kbottenh/Dropbox/Projects/physics-retrieval/data'
roi_dir = '/Users/kbottenh/Dropbox/Data/templates/shen2015/'
fig_dir = '/Users/kbottenh/Dropbox/Projects/physics-retrieval/figures/'

shen = '/Users/kbottenh/Dropbox/Projects/physics-retrieval/shen2015_2mm_268_parcellation.nii.gz'
craddock = '/home/kbott006/physics-retrieval/craddock2012_tcorr05_2level_270_2mm.nii.gz'
masks = ['shen2015', 'craddock2012']

tasks = {'retr': [{'conditions': ['Physics', 'General']},
                  {'runs': [0,1]}], 
         'fci': [{'conditions': ['Physics', 'NonPhysics']},
                  {'runs': [0,1,2]}]}

variables = ['iq', 'iqXsex', 'iqXclass', 'iqXsexXclass', 'sexXclass', 'F', 'Mod', 'Age', 'Strt.Level', 'fd']

sessions = [0,1]

colors = sns.blend_palette(['#ec407a', '#ff7043', '#ffca28',
                            '#d4e157', '#66bb6a', '#26c6da',
                            '#42a5f5', '#7e57c2'], 
                           n_colors=268, as_cmap=True)


# # Data wrangling
# Nodal efficiency data is currently in an <i>incredbily</i> long, multi-indexed dataframe. 
# Here, we transform it into wide data (dataframe per condition per task per session) for ease of analysis later.


df = pd.read_csv(join(data_dir, 'physics-learning-tasks_graphtheory_shen+craddock_nodal.csv'), index_col=0, header=0)
df.rename({'Unnamed: 1': 'session', 'Unnamed: 2': 'task', 'Unnamed: 3': 'condition'}, axis=1, inplace=True)
null_df = pd.read_csv(join(sink_dir, 'local_efficiency', 'task_eff_dist.csv'), 
                      index_col=[0,1,2,3], header=0)


j = list(set(df.columns) - set(['session', 'task', 'condition', 'mask']))
j.sort()
conns = j[268:]

big_df = pd.read_csv(join(data_dir, 'rescored', 'non-brain-data+fd.csv'), index_col=0, header=0)

for mask in masks:
    fci_df = df[df['mask'] == mask]
    fci_df = fci_df[fci_df['task'] == 'fci']
    fci_pre = fci_df[fci_df['session'] == 0]
    fci_pre_phys = fci_pre[fci_pre['condition'] == 'high-level']
    fci_pre_ctrl = fci_pre[fci_pre['condition'] == 'lower-level']
    fci_post = fci_df[fci_df['session'] == 1]
    fci_post_phys = fci_post[fci_post['condition'] == 'high-level']
    fci_post_ctrl = fci_post[fci_post['condition'] == 'lower-level']


    retr_df = df[df['mask'] == mask]
    retr_df = retr_df[retr_df['task'] == 'retr']
    retr_pre = retr_df[retr_df['session'] == 0]
    retr_pre_phys = retr_pre[retr_pre['condition'] == 'high-level']
    retr_pre_ctrl = retr_pre[retr_pre['condition'] == 'lower-level']
    retr_post = retr_df[retr_df['session'] == 1]
    retr_post_phys = retr_post[retr_post['condition'] == 'high-level']
    retr_post_ctrl = retr_post[retr_post['condition'] == 'lower-level']

    fci_pre_phys.drop(['session', 'task', 'condition', 'mask'], axis=1, inplace=True)
    fci_post_phys.drop(['session', 'task', 'condition', 'mask'], axis=1, inplace=True)

    fci_pre_ctrl.drop(['session', 'task', 'condition', 'mask'], axis=1, inplace=True)
    fci_post_ctrl.drop(['session', 'task', 'condition', 'mask'], axis=1, inplace=True)

    retr_pre_phys.drop(['session', 'task', 'condition', 'mask'], axis=1, inplace=True)
    retr_post_phys.drop(['session', 'task', 'condition', 'mask'], axis=1, inplace=True)

    retr_pre_ctrl.drop(['session', 'task', 'condition', 'mask'], axis=1, inplace=True)
    retr_post_ctrl.drop(['session', 'task', 'condition', 'mask'], axis=1, inplace=True)

    for i in np.arange(0,268)[::-1] :
        fci_post_phys.rename({'lEff{0}'.format(i): 'lEff{0}'.format(i+1)}, axis=1, inplace=True)
        fci_pre_phys.rename({'lEff{0}'.format(i): 'lEff{0}'.format(i+1)}, axis=1, inplace=True)
        retr_post_phys.rename({'lEff{0}'.format(i): 'lEff{0}'.format(i+1)}, axis=1, inplace=True)
        retr_pre_phys.rename({'lEff{0}'.format(i): 'lEff{0}'.format(i+1)}, axis=1, inplace=True)

    pre_retr_phys_mean = null_df.loc['pre', 'retr', 'physics', mask]['mean']
    pre_retr_ctrl_mean = null_df.loc['pre', 'retr', 'control', mask]['mean']

    pre_retr_phys_sdev = null_df.loc['pre', 'retr', 'physics', mask]['sdev']
    pre_retr_ctrl_sdev = null_df.loc['pre', 'retr', 'control', mask]['sdev']

    post_retr_phys_mean = null_df.loc['post', 'retr', 'physics', mask]['mean']
    post_retr_ctrl_mean = null_df.loc['post', 'retr', 'control', mask]['mean']

    post_retr_phys_sdev = null_df.loc['post', 'retr', 'physics', mask]['sdev']
    post_retr_ctrl_sdev = null_df.loc['post', 'retr', 'control', mask]['sdev']

    pre_fci_phys_mean = null_df.loc['pre', 'fci', 'physics', mask]['mean']
    pre_fci_ctrl_mean = null_df.loc['pre', 'fci', 'control', mask]['mean']

    pre_fci_phys_sdev = null_df.loc['pre', 'fci', 'physics', mask]['sdev']
    pre_fci_ctrl_sdev = null_df.loc['pre', 'fci', 'control', mask]['sdev']

    post_fci_phys_mean = null_df.loc['post', 'fci', 'physics', mask]['mean']
    post_fci_ctrl_mean = null_df.loc['post', 'fci', 'control', mask]['mean']

    post_fci_phys_sdev = null_df.loc['post', 'fci', 'physics', mask]['sdev']
    post_fci_ctrl_sdev = null_df.loc['post', 'fci', 'control', mask]['sdev']


    #compare null distributions across task condition
    pre_retr_phys_ctrl = (pre_retr_phys_mean - pre_retr_ctrl_mean) / np.sqrt((pre_retr_phys_sdev ** 2) + (pre_retr_ctrl_sdev ** 2))
    print('pre-instruction knowledge, phys > control:\t', pre_retr_phys_ctrl)
    post_retr_phys_ctrl = (post_retr_phys_mean - post_retr_ctrl_mean) / np.sqrt((post_retr_phys_sdev ** 2) + (post_retr_ctrl_sdev ** 2))
    print('post-instruction knowledge, phys > control:\t', post_retr_phys_ctrl)


    pre_fci_phys_ctrl = (pre_fci_phys_mean - pre_fci_ctrl_mean) / np.sqrt((pre_fci_phys_sdev ** 2) + (pre_fci_ctrl_sdev ** 2))
    print('pre-instruction reasoning, phys > control:\t', pre_fci_phys_ctrl)
    post_fci_phys_ctrl = (post_fci_phys_mean - post_fci_ctrl_mean) / np.sqrt((post_fci_phys_sdev ** 2) + (post_fci_ctrl_sdev ** 2))
    print('post-instruction reasoning, phys > control:\t', post_fci_phys_ctrl)

    #compare null distributions across time
    print('\n')
    retr_phys = (post_retr_phys_mean - pre_retr_phys_mean) / np.sqrt((pre_retr_phys_sdev ** 2) + (post_retr_phys_sdev ** 2))
    print('physics knowledge, post > pre:\t', retr_phys)
    retr_ctrl = (post_retr_ctrl_mean - pre_retr_ctrl_mean) / np.sqrt((pre_retr_ctrl_sdev ** 2) + (post_retr_ctrl_sdev ** 2))
    print('general knowledge, post > pre:\t', retr_ctrl)

    fci_phys = (post_fci_phys_mean - pre_fci_phys_mean) / np.sqrt((pre_fci_phys_sdev ** 2) + (post_fci_phys_sdev ** 2))
    print('physics fci, post > pre:\t', fci_phys)
    fci_ctrl = (post_fci_ctrl_mean - pre_fci_ctrl_mean) / np.sqrt((pre_fci_ctrl_sdev ** 2) + (post_fci_ctrl_sdev ** 2))
    print('general fci, post > pre:\t', fci_ctrl)


    #standardize against the empirical null distribution!
    fci_pre_phys[conns] = fci_pre_phys[conns] / null_df.loc['pre', 'fci', 'physics', mask]['mean']
    retr_pre_phys[conns] = retr_pre_phys[conns] / null_df.loc['pre', 'retr', 'physics', mask]['mean']
    #rest_pre[conns] = (rest_pre[conns] - rest_pre[conns].mean()) / rest_pre[conns].std()
    fci_post_phys[conns] = fci_post_phys[conns] / null_df.loc['post', 'fci', 'physics', mask]['mean']
    retr_post_phys[conns] = retr_post_phys[conns] / null_df.loc['post', 'retr', 'physics', mask]['mean']
    #rest_post[conns] = (rest_post[conns] - rest_post[conns].mean()) / rest_post[conns].std()


    effs = {'post phys fci': {'conns': fci_post_phys, 'iqs': ['deltaPRI', 'deltaFSIQ']},
            'post phys retr': {'conns': retr_post_phys, 'iqs': ['WMI2', 'VCI2']}}
    iqs = effs['post phys fci']['iqs'] + effs['post phys retr']['iqs']


    # # Regress local efficiency on IQ and all the covariates
    # Permuted OLS tests each `target_var` independently, while regressing out `confounding_vars`, so to run a multiple regression, we test each variable of interest, separately, and put all other variables in the regression in with the confounds. This way, we can test interactions <i>with</i> main effects.
    # <br><br>
    # Maximum p-values are saved in `sig` dictionary and for each significant variable, the p- and t-values for each node are saved in `nodaleff_sig`.
    # <br><br>
    # For each regression, maximum <i>p</i>- and <i>t</i>-values are stored in `params`, along with nodes whose local efficiency is significantly related to each parameter, are stored <i> by variable</i>.


    sig = {}
    nodaleff_sig = pd.DataFrame(index=conns)
    index = pd.MultiIndex.from_product([iqs, effs.keys(), variables])
    params = pd.DataFrame(index=index, columns=['max nlog(p)', 'max t', 'nodes'])


    for key in effs.keys():
        print(key)
        efficiency = effs[key]['conns']
        iqs = effs[key]['iqs']
        all_data = pd.concat([big_df, efficiency], axis=1)
        all_data.dropna(how='any', axis=0, inplace=True)
        all_data[conns] = (all_data[conns] - all_data[conns].mean()) / all_data[conns].std()
        for iq in iqs:
            print(iq)
            variables = ['{0}'.format(iq), '{0}XSex'.format(iq), '{0}XClass'.format(iq), 
                        '{0}XClassXSex'.format(iq),
                        'F', 'Strt.Level', 'SexXClass', 'Age', 'Mod', '{0} fd'.format(key)]
            for var in variables:
                covariates = list(set(variables) - set([var]))
                p, t, _ = permuted_ols(all_data[var], 
                                    all_data[conns], 
                                    all_data[covariates],
                                    n_perm=10000)
                print(key, var, 'max p-val:',  np.max(p[0]))
                sig['{0}, {1}'.format(iq, key)] = np.max(p[0])
                nodaleff_sig['{0} {1} p'.format(iq, key)] = p.T
                nodaleff_sig['{0} {1} t'.format(iq, key)] = t.T
                sig_nodes = nodaleff_sig[nodaleff_sig['{0} {1} p'.format(iq, key)] >= 1].index
                print('# significant nodes:', len(sig_nodes))
                if key in var:
                    params.loc[iq, key, 'fd']['max nlog(p)'] = np.max(p[0])
                    params.loc[iq, key, 'fd']['max t'] = np.max(t[0])
                    params.loc[iq, key, 'fd']['nodes'] = list(sig_nodes)
                elif iq in var:
                    if 'Sex' in var:
                        if 'Class' in var:
                            params.loc[iq, key, 'iqXsexXclass']['max nlog(p)'] = np.max(p[0])
                            params.loc[iq, key, 'iqXsexXclass']['max t'] = np.max(t[0])
                            params.loc[iq, key, 'iqXsexXclass']['nodes'] = list(sig_nodes)
                        else:
                            params.loc[iq, key, 'iqXsex']['max nlog(p)'] = np.max(p[0])
                            params.loc[iq, key, 'iqXsex']['max t'] = np.max(t[0])
                            params.loc[iq, key, 'iqXsex']['nodes'] = list(sig_nodes)
                    if 'Class' in var:
                        if not 'Sex' in var:
                            params.loc[iq, key, 'iqXclass']['max nlog(p)'] = np.max(p[0])
                            params.loc[iq, key, 'iqXclass']['max t'] = np.max(t[0])
                            params.loc[iq, key, 'iqXclass']['nodes'] = list(sig_nodes)
                    else:
                        params.loc[iq, key, 'iq']['max nlog(p)'] = np.max(p[0])
                        params.loc[iq, key, 'iq']['max t'] = np.max(t[0])
                        params.loc[iq, key, 'iq']['nodes'] = list(sig_nodes)
                elif var == 'SexXClass':
                    params.loc[iq, key, 'sexXclass']['max nlog(p)'] = np.max(p[0])
                    params.loc[iq, key, 'sexXclass']['max t'] = np.max(t[0])
                    params.loc[iq, key, 'sexXclass']['nodes'] = list(sig_nodes)
                else:
                    params.loc[iq, key, var]['max nlog(p)'] = np.max(p[0])
                    params.loc[iq, key, var]['max t'] = np.max(t[0])
                    params.loc[iq, key, var]['nodes'] = list(sig_nodes)


    params.dropna(how='all', inplace=True)


    nodaleff_sig.to_csv(join(sink_dir, '{0}_local_efficiency_iq_sig_all.csv'.format(mask)))
    params.to_csv(join(sink_dir, '{0}_local_efficiency_iq_sig-nodes.csv'.format(mask)))


    n_map = int(len(params[params['max nlog(p)'] > 1].index)) + 1
    interval = 1 / n_map
    husl_pal = sns.husl_palette(n_colors=n_map, h=interval)
    husl_cmap = LinearSegmentedColormap.from_list(husl_pal, husl_pal, N=n_map)
    sns.palplot(husl_pal)

    crayons_l = sns.crayon_palette(['Vivid Tangerine', 'Cornflower'])
    crayons_d = sns.crayon_palette(['Brick Red', 'Midnight Blue'])
    grays = sns.light_palette('#999999', n_colors=3, reverse=True)

    f_2 = sns.crayon_palette(['Red Orange', 'Vivid Tangerine'])
    m_2 = sns.crayon_palette(['Cornflower', 'Cerulean'])


    # In[58]:


    empty_nii = nib.load(join(roi_dir, 'roi101.nii.gz'))
    empty_roi = empty_nii.get_fdata() * 0
    empty = nib.Nifti1Image(empty_roi, empty_nii.affine)
    g = plot_glass_brain(empty, colorbar=False, vmin=0.5, vmax=n_col)
    i = 0


    for var in params.index:
        if params.loc[var]['max nlog(p)'] > 1:
            i += 1
            husl_pal = sns.husl_palette(h=interval * i, n_colors=n_map)
            rois = None
            print(i, var)
            corr_nodes = []
            #tvals = params.loc[i]['max t']
            nodes = params.loc[var]['nodes']
            corr_nodes.append(int(nodes[0].strip('lEff')))
            roi_nifti = nib.load(join(roi_dir,'roi{0}.nii.gz'.format(int(nodes[0].strip('lEff')))))
            roi = roi_nifti.get_fdata()
            rois = (roi * i)
            print(int(nodes[0].strip('lEff')), np.max(rois))
            if len(nodes) > 1:
                for node in nodes[1:]:
                    corr_nodes.append(int(node.strip('lEff')))
                    roi_nifti = nib.load(join(roi_dir,'roi{0}.nii.gz'.format(int(node.strip('lEff')))))
                    roi = roi_nifti.get_fdata()
                    rois += (roi * i)
                    print(int(node.strip('lEff')), np.max(rois))
            else:
                pass
            np.savetxt(join(fig_dir, '{0}-{1}-{2}-.txt'.format(mask, i, var)), corr_nodes, delimiter=',')
            rois_nifti = nib.Nifti1Image(rois, roi_nifti.affine)
            rois_nifti.to_filename(join(data_dir, 'output/local_efficiency', '{0}_nodes.nii.gz'.format(var)))
            h = plot_glass_brain(rois_nifti, cmap=LinearSegmentedColormap.from_list(husl_pal, husl_pal, N=3))
            h.savefig(join(fig_dir, '{0}-{1}-{2}-_ROIs.png'.format(mask, i, var)), dpi=300)
            
            husl_pal = sns.husl_palette(n_colors=int(n_col), h=interval*i)
            g.add_contours(rois_nifti, colors=husl_pal, filled=True, alpha=0.7)
            
        else:
            pass
        
    g.savefig(join(fig_dir, '{0}-LEffXIQ_ROIs.png'.format(mask)), dpi=300)

    n_col = int(len(nodaleff_sig.columns)/2) + 1
    husl_pal = sns.husl_palette(n_colors=int(n_col))
    husl_cmap = LinearSegmentedColormap.from_list(husl_pal, husl_pal, N=int(n_col))
    i = 0
    for var in params.index:
        if params.loc[var]['max nlog(p)'] > 1:
            iq = var[0]
            task = var[1]
            dat = effs[task]['conns']
            husl_pal = sns.husl_palette(h=(interval*i), n_colors=int(n_col))

            print(var, i)
            all_data = pd.concat([big_df, dat[conns]], axis=1)
            all_data.dropna(how='any', axis=0, inplace=True)
            nodes = params.loc[var]['nodes']
            print(nodes)
            for node in nodes:
                if var[-1] == 'iqXsex':
                    #print(iq, 'x Sex', node, nodaleff_sig.at[node,'{0}t'.format(var[:-1])])
                    h = sns.lmplot(iq, node, data=all_data, hue='F', palette=crayons_d)
                    h.savefig(join(fig_dir, '{0}-{1}-{2}-scatter.png'.format(mask, var, node)), dpi=300)
                    plt.close()
                elif var[-1] == 'iqXsexXclass':
                    #print(iq, 'x Sex x Class', node,  nodaleff_sig.at[node,'{0}t'.format(var[:-1])])
                    h = sns.lmplot(iq, node, data=all_data[all_data['F'] == 1], hue='Mod', palette=f_2)
                    h.savefig(join(fig_dir, '{0}-{1}-{2}-scatter-f.png'.format(mask, var, node)), dpi=300)
                    h = sns.lmplot(iq, node, data=all_data[all_data['F'] == 0], hue='Mod', palette=m_2)
                    h.savefig(join(fig_dir, '{0}-{1}-{2}-scatter-m.png'.format(mask, var, node)), dpi=300)
                    plt.close()
                elif var[-1] == 'iqXclass':
                    #print(iq, 'x Class', node,  nodaleff_sig.at[node,'{0}t'.format(column[:-1])])
                    h = sns.lmplot(iq, node, data=all_data, hue='Mod', palette=grays)
                    h.savefig(join(fig_dir, '{0}-{1}-{2}-scatter.png'.format(mask, var, node)), dpi=300)
                    plt.close()
                elif var[-1] == 'sexXclass':
                    #print('Sex x Class', node,  nodaleff_sig.at[node,'{0}t'.format(column[:-1])])
                    h = sns.lmplot('F', node, data=all_data[all_data['F'] == 1], hue='Mod', palette=f_2)
                    h.savefig(join(fig_dir, '{0}-{1}-{2}-scatter-.png'.format(mask, var, node)), dpi=300)
                    plt.close()
                elif var[-1] == 'iq':
                    #print('no interxn', iq, node, nodaleff_sig.at[node,'{0}t'.format(column[:-1])])
                    fig,ax = plt.subplots()
                    sns.regplot(all_data[iq], all_data[node], color=husl_pal[0])
                    sns.despine()
                    plt.tight_layout()
                    fig.savefig(join(fig_dir, '{0}-{1}-{2}-scatter.png'.format(mask, var, node)), dpi=300)
                    plt.close()
                elif var[-1] == 'fd':
                    pass
                else:
                    fig,ax = plt.subplots()
                    sns.regplot(all_data[var[-1]], all_data[node], color=husl_pal[0])
                    sns.despine()
                    plt.tight_layout()
                    fig.savefig(join(fig_dir, '{0}-{1}-{2}-scatter.png'.format(mask, var, node)), dpi=300)
                    plt.close()
            i += 1

# %%
