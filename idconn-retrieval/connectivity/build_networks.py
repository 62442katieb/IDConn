import numpy as np
import pandas as pd
from os import makedirs
from os.path import join, exists
from glob import glob
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import nilearn.datasets

#from .utils import contrast

def atlas_picker(atlas, path=None):
    return atlas, path

def _check_dims(matrix):
    """Raise a ValueError if the input matrix has more than two square.
    Parameters
    ----------
    matrix : numpy.ndarray
        Input array.
    """
    if matrix.ndim != 2:
        raise ValueError('Expected a square matrix, got array of shape'
                         ' {0}.'.format(matrix.shape))

def confounds_merger(confounds):
    """Merges multiple confounds text files.
    Parameters
    ----------
    confounds : list-like
        Filenames of confounds files.
    Returns
    -------
    confounds_file : str
        Filename of merged confounds .tsv file
    """
    arrays = {}
    shape0 = np.empty((len(arrays),))
    shape1 = np.empty((len(arrays),))
    i = 0
    for confound in confounds:
        arrays[confound] = np.genfromtxt(confound, delimiter='\t')
        _check_dims(arrays[confound])
        shape0[i] = arrays[confound].shape[0]
        shape1[i] = arrays[confound].shape[1]
        i += 1
    if shape0[0] > shape1[0]:
        all_conf = np.vstack((arrays[confounds[0]], arrays[confounds[1]]))
        for confound in confounds[1:]:
            all_conf = np.vstack((all_conf, arrays[confound]))
    out_file = join(deriv_dir,
                    subject,
                    session,
                    '{0}_{1}_{2}_{3}_confounds.tsv'.format(subject, session, task, run))
    np.savetxt(out_file, all_conf, delimiter='\t')
    return out_file

def task_networks(dset_dir, subject, session, task, event_related, conditions, runs, connectivity_metric, space, atlas, confounds):
    """
    Makes connectivity matrices per subject per session per task per condition.
    Parameters
    ----------
    dset_dir : str
        BIDS-formatted dataset path (top-level, in which a 'derivatives/' directory will be made if one does not exist)
    subject : str
        Subject ID for which the networks will be calculated.
    session : str, optional
        Session of data collection. If there's only one session, we'll find it.
    task : str
        Task name for which networks will be calculated.
    conditions : list-like
        Conditions of the task for which networks will be separated.
    runs : list-like, optional
        Runs of the task, will be combined to calculate networks.
    connectivity_metric : {"correlation", "partial correlation", "tangent",\
                           "covariance", "precision"}, optional
        The matrix kind. Passed to Nilearn's ConnectivityMeasure.
    space : str
        'native' if analyses will be performed in subjects' functional native space (atlas(es) should be transformed)
        'mni152-2mm' if analyses will be performed in MNI125 2mm isotropic space (fMRI data should already be transformed)
    atlas
    confounds
    confounds : list-like
        Filenames of confounds files.
    Returns
    -------
    confounds_file : str
        Filename of merged confounds .tsv file
    """
    version = '0.1.1'
    if not 'sub' in subject:
        subject = 'sub-{0}'.format(subject)
    if not 'run' in runs[0]:
        for i in np.arange(0, len(runs)):
            runs[i] = 'run-{0}'.format(runs[i])
    if not 'ses' in session:
        session = 'ses-{0}'.format(session)
    if not 'task' in task:
        task = 'task-{0}'.format(task)
    if not 'atlas' in atlas:
        atlas = 'atlas-{0}'.format(atlas)
    subj_dir = join(dset_dir, subject, session, 'func')
    preproc_dir = join(dset_dir, 'derivatives/idconn-preproc', subject, session, 'func')
    deriv_dir = join(dset_dir, 'derivatives/idconn-{0}'.format(version), subject, session, 'func')
    try:
        if not exists(join(dset_dir, 'derivatives/idconn-{0}'.format(version))):
            makedirs(join(dset_dir, 'derivatives/idconn-{0}'.format(version)))
        if not exists(join(dset_dir, 'derivatives/idconn-{0}'.format(version), subject)):
            makedirs(join(dset_dir, 'derivatives/idconn-{0}'.format(version), subject))
        if not exists(join(dset_dir, 'derivatives/idconn-{0}'.format(version), subject, session)):
            makedirs(join(dset_dir, 'derivatives/idconn-{0}'.format(version), subject, session))
        if not exists(deriv_dir):
            makedirs(deriv_dir)
    except Exception as e:
	    print('making dirs error', e)
    run_cond = {}
    for run in runs:
        epi_file = join(preproc_dir,
                        '{0}_{1}_{2}_{3}_bold-mcf.nii.gz'.format(subject, session, task, run))
        assert exists(epi_file), "epi_cleaned does not exist at {0}".format(epi_file)
    
        if space == 'native':
            atlas_file = join(preproc_dir,
                              '{0}_{1}_{2}_{3}_{4}.nii.gz'.format(subject, session, task, run, atlas))
            assert exists(atlas_file), 'atlas/parcellation not found at {0}'.format(atlas)
        else:
            #add in option here to use nilearn-grabbed atlases
            pass
        #LATER: PRINT OVERLAY OF MASK ON EXAMPLE FUNC
        if len(confounds) > 1:
            if not exists(join(preproc_dir,
                               '{0}_{1}_{2}_{3}_bold-confounds+outliers.tsv'.format(subject, session, task, run))):
                #confounds_file = confounds_merger(confounds)
		print('confounds not found')
            else:
                confounds_file = join(preproc_dir,
                                      '{0}_{1}_{2}_{3}_bold-confounds+outliers.tsv'.format(subject, session, task, run))
        else:
            confounds_file = join(preproc_dir,
                                  '{0}_{1}_{2}_{3}_bold-confounds+outliers.tsv'.format(subject, session, task, run))
        #load timing file 
        #update to use pyBIDS + layout
        if exists(join(dset_dir, '{0}_events.tsv'.format(task))):
            run_spec_timing = False
            timing = pd.read_csv(join(dset_dir, '{0}_events.tsv'.format(task)), header=0, index_col=None, sep='\t')
        elif any(task in s for s in glob(join(preproc_dir, '*events.tsv'))):
            run_spec_timing = True
            timing = pd.read_csv(join(preproc_dir, 
                                      '{0}_{1}_{2}_{3}_events.tsv'.format(subject, session, task, run)), sep='\t')
        else:
            print('cannot find task timing file...')
	    timing = None

        if event_related:
            highpass = 1 / 66.
        else:
            highpass = 1 / ((timing.iloc[1]['onset'] - timing.iloc[0]['onset']) * 2)
	    try:        
            #for each parcellation, extract BOLD timeseries
            masker = NiftiLabelsMasker(atlas_file, standardize=True, high_pass=highpass, t_r=2., verbose=1)
            timeseries = masker.fit_transform(epi_file, confounds_file)
            connectivity_measure = ConnectivityMeasure(kind=connectivity_metric)
        except Exception as e:
            print('trying to run masker but', e)
        #load timing file 
        #update to use pyBIDS + layout
	    try:
            #and now we slice into conditions
            for condition in conditions:
                blocks = []
                cond_timing = timing[timing['trial_type'] == condition]
                for i in cond_timing.index:
                    blocks.append((cond_timing.loc[i]['onset'] / 2, ((cond_timing.loc[i]['onset'] + cond_timing.loc[i]['duration']) / 2) + 1))
                if len(blocks) > 1:
                    run_cond['{0}-{1}'.format(run, condition)] = np.vstack((timeseries[int(blocks[0][0]):int(blocks[0][1]), :], timeseries[int(blocks[1][0]):int(blocks[1][1]), :]))
                if len(blocks) > 2:
                    for i in np.arange(2,len(blocks)):
                        run_cond['{0}-{1}'.format(run, condition)] = np.vstack((timeseries[int(blocks[0][0]):int(blocks[0][1]), :], timeseries[int(blocks[1][0]):int(blocks[1][1]), :]))
                    #print('extracted signals for {0}, {1}, {2}'.format(task, run, condition), run_cond['{0}-{1}'.format(run, condition)].shape)
                else:
                    pass
        except Exception as e:
            print('trying to slice and dice, but', e)
    #and paste together the timeseries from each run together per condition
    sliced_ts = {}
    for condition in conditions:
        #print(task, condition, 'pasting timeseries together per condition')
        #print('task isn\'t FCI, only 2 runs')
        #print('{0}-0-{1}'.format(task, condition))
        try:
            if len(runs) > 1:
                sliced_ts[condition] = np.vstack((run_cond['{0}-{1}'.format(runs[0], condition)], run_cond['{0}-{1}'.format(runs[1], condition)]))
                if len(runs) > 2:
                    for run in runs[2:]:
                        sliced_ts[condition] = np.vstack((sliced_ts[condition], run_cond['{0}-{1}'.format(run, condition)]))
            else: 
                sliced_ts[condition] = run_cond['0-{0}'.format(condition)]
            print('{0} total timeseries shape: {1}'.format(condition, sliced_ts[condition].shape))
        except Exception as e:
            print('trying to paste timeseries together, but', e)
        try:
            corrmat = connectivity_measure.fit_transform([sliced_ts[condition]])[0]
            print('{0} corrmat shape: {1}'.format(condition,corrmat.shape))
            corrmat_df = pd.DataFrame(index=np.arange(1, corrmat.shape[0]+1), columns=np.arange(1, corrmat.shape[0]+1),data=corrmat)
            corrmat_file = join(deriv_dir,  
                                '{0}_{1}_{2}-{3}_{4}-corrmat.tsv'.format(subject, session, task, condition, atlas))
        except Exception as e:
            print('trying to make and save corrmat, but', e)
	try:
            corrmat_df.to_csv(corrmat_file, sep='\t')
        except Exception as e:
	    print('saving corrmat...', e)
	return corrmat_file
