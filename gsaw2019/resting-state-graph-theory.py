import numpy as np
import pandas as pd
from os import makedirs
from os.path import join, exists
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import bct
print(pd.__version__)

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

data_dir = '/home/data/nbc/physics-learning/anxiety-physics'
sink_dir = '/home/data/nbc/physics-learning/retrieval-graphtheory'

shen = '/home/kbott006/physics-retrieval/shen2015_2mm_268_parcellation.nii.gz'
craddock = '/home/kbott006/physics-retrieval/craddock2012_tcorr05_2level_270_2mm.nii.gz'
masks = {'shen2015': shen, 'craddock2012': craddock}

sessions = ['pre', 'post']

#shen_masker = NiftiLabelsMasker(shen, background_label=0, standardize=True, detrend=True,t_r=3.)
#craddock_masker = NiftiLabelsMasker(shen, background_label=0, standardize=True, detrend=True,t_r=3.)
#correlation_measure = ConnectivityMeasure(kind='correlation')

index = pd.MultiIndex.from_product([subjects, sessions], names=['subject', 'session'])

df = pd.DataFrame(columns=['shen-efficiency', 'shen-charpath', 'shen-modularity', 'craddock-efficiency', 'craddock-charpath', 'craddock-modularity'], index=index, dtype=np.float64)

for subject in subjects:
    for session in sessions:
        try:
            #confounds = join(data_dir, 'output', session, subject, '{0}_confounds.txt'.format(subject))
            #epi_data = join(data_dir, session, '{0}_filtered_func_data_mni.nii.gz'.format(subject))

            #shen_ts = shen_masker.fit_transform(epi_data, confounds)
            #shen_corrmat = correlation_measure.fit_transform([shen_ts])[0]
            #np.savetxt(join(sink_dir, session, 'resting-state', subject, '{0}_network_corrmat_shen2015.csv'.format(subject)), shen_corrmat, delimiter=",")
            shen_corrmat = np.genfromtxt(join(sink_dir, session, 'resting-state', subject, '{0}_network_corrmat_shen2015.csv'.format(subject)), delimiter=",")

            #craddock_ts = craddock_masker.fit_transform(epi_data, confounds)
            #craddock_corrmat = correlation_measure.fit_transform([craddock_ts])[0]
            #np.savetxt(join(sink_dir, session, 'resting-state', subject, '{0}_network_corrmat_craddock2012.csv'.format(subject)), craddock_corrmat, delimiter=",")
            craddock_corrmat = np.genfromtxt(join(sink_dir, session, 'resting-state', subject, '{0}_network_corrmat_craddock2012.csv'.format(subject)), delimiter=",")

            ge_s = []
            ge_c = []
            cp_s = []
            cp_c = []
            md_s = []
            md_c = []
            for p in np.arange(0.1, 1, 0.1):
                ntwk = []
                shen_thresh = bct.threshold_proportional(shen_corrmat, p, copy=True)
                craddock_thresh = bct.threshold_proportional(craddock_corrmat, p, copy=True)
                #network measures of interest here

                #global efficiency
                ge = bct.efficiency_wei(shen_thresh)
                ge_s.append(ge)
                ge = bct.efficiency_wei(craddock_thresh)
                ge_c.append(ge)

                #characteristic path length
                cp = bct.charpath(shen_thresh)
                cp_s.append(cp[0])
                cp = bct.charpath(craddock_thresh)
                cp_c.append(cp[0])

                #modularity
                md = bct.modularity_louvain_und(shen_thresh)
                md_s.append(md[1])
                md = bct.modularity_louvain_und(craddock_thresh)
                md_c.append(md[1])
            df.loc[(int(subject), session), 'shen-efficiency'] = np.trapz(ge_s, dx=0.1)
            df.loc[(int(subject), session), 'shen-charpath'] = np.trapz(cp_s, dx=0.1)
            df.loc[(int(subject), session), 'shen-modularity'] = np.trapz(md_s, dx=0.1)
            df.loc[(int(subject), session), 'craddock-efficiency'] = np.trapz(ge_c, dx=0.1)
            df.loc[(int(subject), session), 'craddock-charpath'] = np.trapz(cp_c, dx=0.1)
            df.loc[(int(subject), session), 'craddock-modularity'] = np.trapz(ge_c, dx=0.1)
            #df.to_csv(join(sink_dir, 'resting-state_graphtheory_shen+craddock.csv'), sep=',')
        except Exception as e:
            print(e, subject, session)
        df.to_csv(join(sink_dir, 'resting-state_graphtheory_shen+craddock.csv'), sep=',')

df.to_csv(join(sink_dir, 'resting-state_graphtheory_shen+craddock.csv'), sep=',')
