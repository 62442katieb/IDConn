import numpy as np
import pandas as pd
from os import makedirs
from os.path import join, exists
from nipype.interfaces.fsl import InvWarp, ApplyWarp
from nilearn import plotting

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

data_dir = '/home/data/nbc/physics-learning/data/pre-processed/'
sink_dir = '/home/data/nbc/physics-learning/retrieval-graphtheory/'

shen = '/home/kbott006/physics-retrieval/shen2015_2mm_268_parcellation.nii.gz'
craddock = '/home/kbott006/physics-retrieval/craddock2012_tcorr05_2level_270_2mm.nii.gz'
masks = {'shen2015': shen, 'craddock2012': craddock}

sessions = ['pre', 'post']

for subject in subjects:
    for session in sessions:
        if not exists(join(sink_dir, session, 'resting-state', subject)):
            makedirs(join(sink_dir, session, 'resting-state', subject))
        try:
            if session == 'pre':
                example_func = join(data_dir, subject, 'session-0', 'resting-state', 'resting-state-0', 'endor1.feat', 'example_func.nii.gz')
                func2mni_warp = join(data_dir, subject, 'session-0', 'resting-state', 'resting-state-0', 'endor1.feat', 'reg', 'example_func2standard_warp.nii.gz')

            if session == 'pre':
                example_func = join(data_dir, subject, 'session-1', 'resting-state', 'resting-state-0', 'endor1.feat', 'example_func.nii.gz')
                func2mni_warp = join(data_dir, subject, 'session-1', 'resting-state', 'resting-state-0', 'endor1.feat', 'reg', 'example_func2standard_warp.nii.gz')

            #invwarp = InvWarp()
            #invwarp.inputs.warp = func2mni_warp
            #invwarp.inputs.reference = example_func
            #invwarp.inputs.output_type = "NIFTI_GZ"
            #invwarp.inputs.inverse_warp = join(sink_dir, session, 'resting-state', subject, '{0}-{1}_mni-fnirt-epi-warp.nii.gz'.format(subject, session))

            #rev = invwarp.run()
            for key in masks.keys():
                applywarp = ApplyWarp(interp="nn", abswarp=True)
                applywarp.inputs.in_file = masks[key]
                applywarp.inputs.ref_file = example_func
                applywarp.inputs.field_file = join(sink_dir, session, 'resting-state', subject, '{0}-{1}_mni-fnirt-epi-warp.nii.gz'.format(subject, session))
                applywarp.inputs.out_file = join(sink_dir, session, 'resting-state', subject, '{0}-{1}_{2}.nii.gz'.format(subject, session, key))
                res = applywarp.run()

                qa_file = join(sink_dir, 'output', 'qa', '{0}-{1}_qa-rest_{2}.png'.format(subject, session, key))

                display = plotting.plot_roi(res.outputs.out_file, bg_img=example_func,
                                            colorbar=True, vmin=0, vmax=18,
                                            draw_cross=False)
                display.savefig(qa_file, dpi=300)
                display.close()
        except Exception as e:
            print(e)
