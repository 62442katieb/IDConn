from nilearn import plotting
from nipype.interfaces.fsl import FLIRT, FNIRT
from nipype.interfaces.utility import Function
from os.path import join

def flirt_t1_epi(subject_id, data_dir, run, sink_dir):
    def get_niftis(subject_id, data_dir, run, sink_dir):
        from os.path import join, exists
        t1 = join(data_dir, subject_id, 'session-1', 'anatomical', 'anatomical-0', 'anatomical.nii.gz')
        #t1_brain_mask = join(data_dir, subject, 'session-1', 'anatomical', 'anatomical-0', 'fsl', 'anatomical-bet.nii.gz')
        ex_func = join(sink_dir, subject_id,'{0}-{1}_retr-example_func.nii.gz'.format(subject_id, run))
        assert exists(t1), "t1 does not exist"
        assert exists(ex_func), "example func does not exist"
        standard = '/home/applications/fsl/5.0.8/data/standard/MNI152_T1_2mm.nii.gz'
        return t1, ex_func, standard

    coreg = join(sink_dir, 'qa', '{0}-{1}_t1_flirt.png'.format(subject,run))

    data = Function(function=get_niftis, input_names=["subject_id", "data_dir", "run", "sink_dir"],
                            output_names=["t1", "ex_func", "standard"])
    data.inputs.data_dir = data_dir
    data.inputs.sink_dir = sink_dir
    data.inputs.subject_id = subject
    data.inputs.run = run
    grabber = data.run()

    re_flit = FLIRT(cost_func="normmi", dof=12, searchr_x=[-90,90],
                    searchr_y=[-90,90], searchr_z=[-90,90], interp='trilinear', bins=256)
    re_flit.inputs.reference = grabber.outputs.ex_func
    re_flit.inputs.in_file = grabber.outputs.t1
    re_flit.inputs.out_file=join(sink_dir, subject, '{0}-{1}_t1-flirt-retr.nii.gz'.format(subject,run))
    re_flit.inputs.out_matrix_file = join(sink_dir, subject, '{0}-{1}_t1-flirt-retr.mat'.format(subject,run))
    reg_func = re_flit.run()

    display = plotting.plot_anat(grabber.outputs.ex_func, dim=0)
    display.add_edges(reg_func.outputs.out_file)
    display.savefig(coreg, dpi=300)
    return

#subjects whose coregistration failed for both runs
subjects = {'325': '0',
            '102': '1'}

data_dir = '/home/data/nbc/physics-learning/data/pre-processed'
sink_dir = '/home/data/nbc/physics-learning/retrieval-graphtheory/output'

for subject in subjects.keys():
    for run in subjects[subject]:
        flirt_t1_epi(subject, data_dir, run, sink_dir)

#for subject in norm_subjects.keys():
#    for run in subjects[subject]:
#        norm = join(sink_dir, 'qa', '{0}-{1}_mni_fnirt_t1.png'.format(subject,run))

#        data = Function(function=get_niftis, input_names=["subject_id", "data_dir", "run", "sink_dir"],
#                                output_names=["t1", "ex_func", "standard"])
#        data.inputs.data_dir = data_dir
#        data.inputs.sink_dir = sink_dir
#        data.inputs.subject_id = subject
#        data.inputs.run = run
#        grabber = data.run()

#        perf = FNIRT(output_type='NIFTI_GZ')
#        perf.inputs.warped_file = join(sink_dir, subject, '{0}-{1}_mni-fnirt-t1.nii.gz'.format(subject, run))
#        perf.inputs.affine_file = join(sink_dir, subject, '{0}-{1}_mni-flirt-t1.mat'.format(subject, run))
#        perf.inputs.in_file = grabber.outputs.standard
#        perf.inputs.subsampling_scheme = [8,4,2,2]
#        perf.inputs.fieldcoeff_file = join(sink_dir, subject, '{0}-{1}_mni-fnirt-t1-warpcoef.nii.gz'.format(subject, run))
#        perf.inputs.ref_file = grabber.outputs.t1
#        reg2 = perf.run()

#        display = plotting.plot_anat(grabber.outputs.t1, dim=0)
#        display.add_edges(reg2.outputs.warped_file)
#        display.savefig(norm, dpi=300)
