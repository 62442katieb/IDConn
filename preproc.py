from __future__ import division
from os.path import join, basename, exists
from os import makedirs

from nilearn import input_data, datasets, plotting, regions
from nilearn.image import concat_imgs
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from scipy.stats import pearsonr

import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
import nipype.interfaces.utility as util
from nipype.interfaces.fsl import InvWarp, ApplyWarp

import bct
import json
import numpy as np
import pandas as pd
import datetime


# ## Preprocessing
# Largely following the Westphal et al. (2017) paper, but taking into account the things that Dani Bassett does in her papers (which I still need to look into).
# ### Preprocessing methods per Westphal et al., 2017
# 1. Slice timing correction
# 2. Motion correction
# 3. Unwarping
# 4. Coregistration to subject's T1
# 5. Anatomical segmentation
# 6. Spatial normalization to MNI template
# 7. Spatial smoothing (6mm FWHM)
# 8. High-pass filtering (236_s_)
# 9. Timecourse per voxel demeaned.
# ### Alterations made below
# Preprocessing was done with FSL tools in Nipype.
# 3. No fieldmaps, so no unwarping... (look into this)
# 7. No smoothing
# 8. High pass filtering at 55s
# 9. Standardized TS

# In[1]:


def preproc(data_dir, sink_dir, subject, task, session, run, masks, motion_thresh, moco):
    from nipype.interfaces.fsl import MCFLIRT, FLIRT, FNIRT, ExtractROI, ApplyWarp, MotionOutliers, InvWarp, FAST
    #from nipype.interfaces.afni import AlignEpiAnatPy
    from nipype.interfaces.utility import Function
    from nilearn.plotting import plot_anat
    from nilearn import input_data

    #WRITE A DARA GRABBER
    def get_niftis(subject_id, data_dir, task, run, session):
        from os.path import join, exists
        t1 = join(data_dir, subject_id, 'session-{0}'.format(session), 'anatomical', 'anatomical-0', 'anatomical.nii.gz')
        #t1_brain_mask = join(data_dir, subject_id, 'session-1', 'anatomical', 'anatomical-0', 'fsl', 'anatomical-bet.nii.gz')
        epi = join(data_dir, subject_id, 'session-{0}'.format(session), task, '{0}-{1}'.format(task, run), '{0}.nii.gz'.format(task))
        assert exists(t1), "t1 does not exist at {0}".format(t1)
        assert exists(epi), "epi does not exist at {0}".format(epi)
        standard = '/home/applications/fsl/5.0.8/data/standard/MNI152_T1_2mm.nii.gz'
        return t1, epi, standard

    data = Function(function=get_niftis, input_names=["subject_id", "data_dir", "task", "run", "session"],
                            output_names=["t1", "epi", "standard"])
    data.inputs.data_dir = data_dir
    data.inputs.subject_id = subject
    data.inputs.run = run
    data.inputs.session = session
    data.inputs.task = task
    grabber = data.run()

    if session == 0:
        sesh = 'pre'
    if session == 1:
        sesh = 'post'

    #reg_dir = '/home/data/nbc/physics-learning/data/first-level/{0}/session-1/retr/retr-{1}/retr-5mm.feat/reg'.format(subject, run)
    #set output paths for quality assurance pngs
    qa1 = join(sink_dir, 'qa', '{0}-session-{1}_{2}-{3}_t1_flirt.png'.format(subject, session, task, run))
    qa2 = join(sink_dir, 'qa', '{0}-session-{1}_{2}-{3}_mni_flirt.png'.format(subject, session, task, run))
    qa3 = join(sink_dir, 'qa', '{0}-session-{1}_{2}-{3}_mni_fnirt.png'.format(subject, session, task, run))
    confound_file = join(sink_dir, subject,'{0}-session-{1}_{2}-{3}_confounds.txt'.format(subject, session, task, run))

    #run motion correction if indicated
    if moco == True:
        mcflirt = MCFLIRT(ref_vol=144, save_plots=True, output_type='NIFTI_GZ')
        mcflirt.inputs.in_file = grabber.outputs.epi
        #mcflirt.inputs.in_file = join(data_dir, subject, 'session-1', 'retr', 'retr-{0}'.format(run), 'retr.nii.gz')
        mcflirt.inputs.out_file = join(sink_dir, sesh, subject,'{0}-session-{1}_{2}-{3}_mcf.nii.gz'.format(subject, session, task, run))
        flirty = mcflirt.run()
        motion = np.genfromtxt(flirty.outputs.par_file)
    else:
        print "no moco needed"
        motion = 0

    #calculate motion outliers
    try:
        mout = MotionOutliers(metric='fd', threshold=motion_thresh)
        mout.inputs.in_file = grabber.outputs.epi
        mout.inputs.out_file = join(sink_dir, sesh, subject, '{0}-session-{1}_{2}-{3}_fd-gt-{3}mm'.format(subject, session, task, run, motion_thresh))
        mout.inputs.out_metric_plot = join(sink_dir, sesh, subject, '{0}-session-{1}_{2}-{3}_metrics.png'.format(subject, session, task, run))
        mout.inputs.out_metric_values = join(sink_dir, sesh, subject,'{0}-session-{1}_{2}-{3}_fd.txt'.format(subject, session, task, run))
        moutliers = mout.run()
        outliers = np.genfromtxt(moutliers.outputs.out_file)
        e = 'no errors in motion outliers, yay'
    except Exception as e:
        print(e)
        outliers = np.genfromtxt(mout.inputs.out_metric_values)
        #set everything above the threshold to 1 and everything below to 0
        outliers[outliers > motion_thresh] = 1
        outliers[outliers < motion_thresh] = 0

    #concatenate motion parameters and motion outliers to form confounds file

    #outliers = outliers.reshape((outliers.shape[0],1))
    conf = outliers
    np.savetxt(confound_file, conf, delimiter=',')

    #extract an example volume for normalization
    ex_fun = ExtractROI(t_min=144, t_size=1)
    ex_fun.inputs.in_file = flirty.outputs.out_file
    ex_fun.inputs.roi_file = join(sink_dir, sesh, subject,'{0}-session-{1}_{2}-{3}-example_func.nii.gz'.format(subject, session, task, run))
    fun = ex_fun.run()



    warp = ApplyWarp(interp="nn", abswarp=True)

    if not exists('/home/data/nbc/physics-learning/data/first-level/{0}/session-{1}/{2}/{2}-{3}/{2}-5mm.feat/reg/example_func2standard_warp.nii.gz'.format(subject, session, task, run)):
        #two-step normalization using flirt and fnirt, outputting qa pix
        flit = FLIRT(cost_func="corratio", dof=12)
        reg_func = flit.run(reference=fun.outputs.roi_file, in_file=grabber.outputs.t1, searchr_x=[-180,180], searchr_y=[-180,180],
                            out_file=join(sink_dir, sesh, subject, '{0}-session-{1}_{2}-{3}_t1-flirt.nii.gz'.format(subject, session, task, run)),
                            out_matrix_file = join(sink_dir, sesh, subject, '{0}-session-{1}_{2}-{3}_t1-flirt.mat'.format(subject, session, task, run)))
        reg_mni = flit.run(reference=grabber.outputs.t1, in_file=grabber.outputs.standard, searchr_y=[-180,180], searchr_z=[-180,180],
                            out_file=join(sink_dir, sesh, subject, '{0}-session-{1}_{2}-{3}_mni-flirt-t1.nii.gz'.format(subject, session, task, run)),
                            out_matrix_file = join(sink_dir, sesh, subject, '{0}-session-{1}_{2}-{3}_mni-flirt-t1.mat'.format(subject, session, task, run)))

        #plot_stat_map(aligner.outputs.out_file, bg_img=fun.outputs.roi_file, colorbar=True, draw_cross=False, threshold=1000, output_file=qa1a, dim=-2)
        display = plot_anat(fun.outputs.roi_file, dim=-1)
        display.add_edges(reg_func.outputs.out_file)
        display.savefig(qa1, dpi=300)
        display.close()

        display = plot_anat(grabber.outputs.t1, dim=-1)
        display.add_edges(reg_mni.outputs.out_file)
        display.savefig(qa2, dpi=300)
        display.close()

        perf = FNIRT(output_type='NIFTI_GZ')
        perf.inputs.warped_file = join(sink_dir, sesh, subject, '{0}-session-{1}_{2}-{3}_mni-fnirt-t1.nii.gz'.format(subject, session, task, run))
        perf.inputs.affine_file = reg_mni.outputs.out_matrix_file
        perf.inputs.in_file = grabber.outputs.standard
        perf.inputs.subsampling_scheme = [8,4,2,2]
        perf.inputs.fieldcoeff_file = join(sink_dir, sesh, subject, '{0}-session-{1}_{2}-{3}_mni-fnirt-t1-warpcoeff.nii.gz'.format(subject, session, task, run))
        perf.inputs.field_file = join(sink_dir, sesh, subject, '{0}-session-{1}_{2}-{3}_mni-fnirt-t1-warp.nii.gz'.format(subject, session, task, run))
        perf.inputs.ref_file = grabber.outputs.t1
        reg2 = perf.run()
        warp.inputs.field_file = reg2.outputs.field_file
        #plot fnirted MNI overlaid on example func
        display = plot_anat(grabber.outputs.t1, dim=-1)
        display.add_edges(reg2.outputs.warped_file)
        display.savefig(qa3, dpi=300)
        display.close()
    else:
        warpspeed = InvWarp(output_type='NIFTI_GZ')
        warpspeed.inputs.warp = '/home/data/nbc/physics-learning/data/first-level/{0}/session-{1}/{2}/{2}-{3}/{2}-5mm.feat/reg/example_func2standard_warp.nii.gz'.format(subject, session, task, run)
        warpspeed.inputs.reference = fun.outputs.roi_file
        warpspeed.inputs.inverse_warp = join(sink_dir, sesh, subject, '{0}-session-{1}_{2}-{3}_mni-fnirt-t1-warp.nii.gz'.format(subject, session, task, run))
        mni2epiwarp = warpspeed.run()
        warp.inputs.field_file = mni2epiwarp.outputs.inverse_warp

    for key in masks.keys():
        #warp takes us from mni to epi
        warp.inputs.in_file = masks[key]
        warp.inputs.ref_file = fun.outputs.roi_file
        warp.inputs.out_file = join(sink_dir, sesh, subject,'{0}-session-{1}_{2}-{3}_{4}.nii.gz'.format(subject, session, task, run, key))
        net_warp = warp.run()

        qa_file = join(sink_dir, 'qa', '{0}-session-{1}_{2}-{3}_qa_{4}.png'.format(subject, session, task, run, key))

        display = plotting.plot_roi(net_warp.outputs.out_file, bg_img=fun.outputs.roi_file,
                                    colorbar=True, vmin=0, vmax=18,
                                    draw_cross=False)
        display.savefig(qa_file, dpi=300)
        display.close()

    return flirty.outputs.out_file, confound_file, e

#choose your atlas and either fetch it from Nilearn using one of the the 'datasets' functions
shen = '/home/kbott006/physics-retrieval/shen2015_2mm_268_parcellation.nii.gz'
craddock = '/home/kbott006/physics-retrieval/craddock2012_tcorr05_2level_270_2mm.nii.gz'
masks = {'shen2015': shen, 'craddock2012': craddock}


# In[ ]:


#only want post subjects
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
#all subjects 102 103 101 104 106 107 108 110 212 X213 214 215 216 217 218 219 320 321 X322 323 324 325
#327 328 X329 330 331 X332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 451
#X452 453 455 X456 X457 458 459 460 462 463 464 465 467 468 469 470 502 503 571 572 573 574 X575 577 578
#X579 X580 581 582 584 585 586 587 588 589 X590 591 592 593 594 595 596 597 598 604 605 606 607 608 609
#610 X611 612 613 614 615 X616 617 618 619 620 621 622 623 624 625 626 627 X628 629 630 631 633 634
#errors in fnirt-to-mni: 213, 322, 329, 332, 452, 456, 457, 575, 579, 580, 590, 611, 616, 628
#subjects without post-IQ measure: 452, 461, 501, 575, 576, 579, 583, 611, 616, 628, 105, 109, 211, 213, 322, 326, 329, 332

#subjects for whom preproc didn't run because of motion reasons
#subjects_re = {'217': [0], '334': [1], '335': [1], '453': [1], '463': [0,1], '618': [1], '626': [0]}

data_dir = '/home/data/nbc/physics-learning/data/pre-processed'
sink_dir = '/home/data/nbc/physics-learning/retrieval-graphtheory/output'
lab_notebook_dir = '/home/kbott006/lab_notebook/'

motion_thresh=0.9

runs = [0, 1]
sessions = [0,1]
tasks = ['reas', 'retr']
sesh = ['pre', 'post']

index = pd.MultiIndex.from_product([subjects, tasks, sessions], names=['subject', 'task', 'session'])
lab_notebook = pd.DataFrame(index=index, columns=['start', 'end', 'errors'])


#run preprocessing once per run per subject
#for subject in subjects_re.keys():
for subject in subjects:
    if not exists(join(sink_dir, subject)):
        makedirs(join(sink_dir, subject))
    for task in tasks:
        for session in sessions:
            for run in runs:
                lab_notebook.at[(subject, task, session),'start'] = str(datetime.datetime.now())
                #xfm laird 2011 maps to subject's epi space & define masker
                if not exists(join(sink_dir, sesh[session], subject,'{0}-session-{1}_{2}-{3}_mcf.nii.gz'.format(subject, session, task, run))):
                    try:
                        x = preproc(data_dir, sink_dir, subject, task, session, run, masks, motion_thresh, moco=True)
                        lab_notebook.at[(subject, task, session),'end'] = str(datetime.datetime.now())
                    except Exception as e:
                        lab_notebook.at[(subject, task, session),'errors'] = [x[2], e, str(datetime.datetime.now())]
                        print(subject, session, task, run, e, x[2])
                else:
                    lab_notebook.at[(subject, task, session),'errors'] = 'preprocessing already done'

lab_notebook.to_csv(join(lab_notebook_dir, 'phys-iq-preproc_{0}.csv'.format(str(datetime.datetime.now()))))
