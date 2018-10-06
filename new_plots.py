from os.path import join
from nilearn import plotting
from nipype.interfaces.utility import Function
from nipype.interfaces.fsl import ApplyWarp, FNIRT

def fnirt_again(data_dir, sink_dir, subject, run, masks, mask_names):
    import numpy as np
    def get_niftis(subject, data_dir, sink_dir, run):
        from os.path import join, exists
        t1 = join(data_dir, subject, 'session-1', 'anatomical', 'anatomical-0', 'anatomical.nii.gz')
        #t1_brain_mask = join(data_dir, subject, 'session-1', 'anatomical', 'anatomical-0', 'fsl', 'anatomical-bet.nii.gz')
        example_func = join(sink_dir, subject,'{0}-{1}_retr-example_func.nii.gz'.format(subject, run))
        assert exists(t1), "t1 does not exist"
        assert exists(example_func), "example_func does not exist"
        standard = '/home/applications/fsl/5.0.8/data/standard/MNI152_T1_2mm.nii.gz'
        mni2t1 = join(sink_dir, subject, '{0}-{1}_mni-flirt-t1.mat'.format(subject, run))
        t12epi = join(sink_dir, subject, '{0}-{1}_t1-flirt-retr.mat'.format(subject, run))
        return t1, example_func, standard, mni2t1, t12epi

    data = Function(function=get_niftis, input_names=["subject", "data_dir", "sink_dir", "run"],
                            output_names=["t1", "example_func", "standard", "mni2t1", "t12epi", ])
    data.inputs.data_dir = data_dir
    data.inputs.sink_dir = sink_dir
    data.inputs.subject = subject
    data.inputs.run = run
    grabber = data.run()

    perf = FNIRT(output_type='NIFTI_GZ')
    perf.inputs.warped_file = join(sink_dir, subject, '{0}-{1}_mni-fnirt-t1.nii.gz'.format(subject, run))
    perf.inputs.affine_file = grabber.outputs.mni2t1
    perf.inputs.in_file = grabber.outputs.standard
    perf.inputs.subsampling_scheme = [8,4,2,2]
    perf.inputs.fieldcoeff_file = join(sink_dir, subject, '{0}-{1}_mni-fnirt-t1-warpcoef.nii.gz'.format(subject, run))
    perf.inputs.field_file = join(sink_dir, subject, '{0}-{1}_mni-fnirt-t1-warp.nii.gz'.format(subject, run))
    perf.inputs.ref_file = grabber.outputs.t1
    reg2 = perf.run()

    for i in np.arange(0, len(masks)):
        #warp takes us from mni to t1, postmat
        warp = ApplyWarp(interp="nn", abswarp=True)
        warp.inputs.in_file = masks[i]
        warp.inputs.ref_file = grabber.outputs.example_func
        warp.inputs.field_file = reg2.outputs.field_file
        warp.inputs.postmat = grabber.outputs.t12epi
        warp.inputs.out_file = join(sink_dir, subject,'{0}-{1}_{2}_retr.nii.gz'.format(subject, run, mask_names[i]))
        net_warp = warp.run()
    return 'yay'

subjects = ['101', '102', '103', '104', '106', '107', '108', '110', '212',
            '214', '215', '216', '217', '218', '219', '320', '321', '323',
            '324', '325', '327', '328', '330', '331', '333', '334',
            '335', '336', '337', '338', '339', '340', '341', '342', '343', '344',
            '345', '346', '347', '348', '349', '350', '451', '453', '455',
            '458', '459', '460', '462', '463', '464', '465', '467',
            '468', '469', '470', '502', '503', '571', '572', '573', '574',
            '577', '578', '581', '582', '584', '585', '586', '587',
            '588', '589', '591', '592', '593', '594', '595', '596', '597',
            '598', '604', '605', '606', '607', '608', '609', '610', '612',
            '613', '614', '615', '617', '618', '619', '620', '621', '622',
            '623', '624', '625', '626', '627', '629', '630', '631', '633',
            '634']
subjects = ['216', '347', '571', '594', '621']

#subjects = ['101']

data_dir = '/home/data/nbc/physics-learning/data/pre-processed'
sink_dir = '/home/data/nbc/physics-learning/retrieval-graphtheory/output'
runs = [0, 1]

laird_2011_icns = '/home/data/nbc/physics-learning/retrieval-graphtheory/18-networks-5.14-mni_2mm.nii.gz'
harvox_hippo = '/home/data/nbc/physics-learning/retrieval-graphtheory/harvox-hippo-prob50-2mm.nii.gz'
masks = [laird_2011_icns, harvox_hippo]
mask_names = ['18_icn', 'hippo']

for subject in subjects:
    for run in runs:

        try:
            #t1 = join(data_dir, subject, 'session-1', 'anatomical', 'anatomical-0', 'anatomical.nii.gz')
            #t1_func = join(sink_dir, subject, '{0}-{1}_t1-flirt-retr.nii.gz'.format(subject, run))
            example_func = join(sink_dir, subject,'{0}-{1}_retr-example_func.nii.gz'.format(subject, run))
            #mni_flirt = join(sink_dir, subject, '{0}-{1}_mni-flirt-t1.nii.gz'.format(subject, run))
            #mni_fnirt = join(sink_dir, subject, '{0}-{1}_mni-fnirt-t1.nii.gz'.format(subject, run))

            coreg = join(sink_dir, 'qa', '{0}-{1}_t1_flirt.png'.format(subject, run))
            norm1 = join(sink_dir, 'qa', '{0}-{1}_mni_flirt.png'.format(subject, run))
            norm2 = join(sink_dir, 'qa', '{0}-{1}_mni_fnirt.png'.format(subject, run))

            #display = plotting.plot_anat(t1, dim=-1)
            #display.add_edges(mni_flirt)
            #display.savefig(norm1, dpi=300)

            #display = plotting.plot_anat(t1, dim=-1)
            #display.add_edges(mni_fnirt)
            #display.savefig(norm2, dpi=300)

            #display = plotting.plot_anat(example_func, dim=0)
            #display.add_edges(t1_func)
            #display.savefig(coreg, dpi=300)

            for mask in mask_names:
                mask_file = join(sink_dir, subject,'{0}-{1}_{2}_retr.nii.gz'.format(subject, run, mask))
                mask_qa = join(sink_dir, 'qa', '{0}-{1}_qa_{2}.png'.format(subject, run, mask))
                display = plotting.plot_roi(mask_file, bg_img=example_func,
                                            colorbar=True, vmin=0, vmax=18,
                                            draw_cross=False)
                display.savefig(mask_qa, dpi=300)
        except Exception as e:
            print e
