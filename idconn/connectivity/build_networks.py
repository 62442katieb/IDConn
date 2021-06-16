from posixpath import sep
import numpy as np
import pandas as pd
import nibabel as nib
import bids
from os import makedirs
from os.path import join, exists, basename
from glob import glob
from nilearn import input_data, datasets, connectome, image, plotting

#from .utils import contrast

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
    """
    DEFUNCT - made obsolete by fmriprep
    Merges multiple confounds text files.
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
        Name of task fMRI scan from which networks will be calculated.
    conditions : list-like
        Conditions of the task for which networks will be separated.
    runs : list-like, optional
        Runs of the task, will be combined to calculate networks.
    connectivity_metric : {"correlation", "partial correlation", "tangent",\
                           "covariance", "precision"}, optional
        The matrix kind. Passed to Nilearn's `ConnectivityMeasure`.
    space : str
        'native' if analyses will be performed in subjects' functional native space (atlas(es) should be transformed)
        'mni152-2mm' if analyses will be performed in MNI125 2mm isotropic space (fMRI data should already be transformed)
    atlas : str
        If you want to grab an atlas using Nilearn, this is the name of the atlas and 
        must match the corresponding function `fetch_atlas_[name]` in `nilearn.datasets`. 
        If you have your own atlas, this is the path to that nifti file.`
    atlas_key : str, optional
        If grabbing an atlas from Nilearn, key that corresponds to a specific version of the atlas specified with `atlas`.
    confounds : list-like
        Filenames of confounds files.
    Returns
    -------
    confounds_file : str
        Filename of merged confounds .tsv file
    """
    #version = '0.1.1'
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
    if not 'space' in space:
        space = 'space-{0}'.format(space)
    subj_dir = join(dset_dir, subject, session, 'func')
    preproc_dir = join(dset_dir, 'derivatives/idconn/preproc', subject, session, 'func') #FIX THIS! SHOULD BE FMRIPREP DIR
    deriv_dir = join(dset_dir, 'derivatives/idconn/{0}'.format(version), subject, session, 'func')
    try:
        if not exists(join(dset_dir, 'derivatives/idconn/{0}'.format(version))):
            makedirs(join(dset_dir, 'derivatives/idconn/{0}'.format(version)))
        if not exists(join(dset_dir, 'derivatives/idconn/{0}'.format(version), subject)):
            makedirs(join(dset_dir, 'derivatives/idconn/{0}'.format(version), subject))
        if not exists(join(dset_dir, 'derivatives/idconn/{0}'.format(version), subject, session)):
            makedirs(join(dset_dir, 'derivatives/idconn/{0}'.format(version), subject, session))
        if not exists(deriv_dir):
            makedirs(deriv_dir)
    except Exception as e:
	    print('making dirs error', e)
    run_cond = {}
    for run in runs:
        bold_file = join(fmriprep_dir, subject, session, 'func', '{0}_{1}_{2}_{3}_{4}_desc-preproc_bold.nii.gz'.format(subject, session, task, run, space))
        assert exists(bold_file), "preprocessed bold file does not exist at {0}".format(bold_file)
        if space == 'native':
            atlas_file = join(preproc_dir,
                              '{0}_{1}_{2}_{3}_{4}.nii.gz'.format(subject, session, task, run, atlas))
            assert exists(atlas_file), 'atlas/parcellation not found at {0}'.format(atlas)
        elif 'mni' in space:
            atlas, atlas_file, dim = atlas_picker(atlas, path=join(dset_dir, 'derivatives/idconn/{0}'.format(version)), key=key)
            #add in option here to use nilearn-grabbed atlases
        elif 'MNI' in space:
            atlas, atlas_file, dim = atlas_picker(atlas, path=join(dset_dir, 'derivatives/idconn/{0}'.format(version)), key=key)
        #LATER: PRINT OVERLAY OF MASK ON EXAMPLE FUNC
        confounds_json = join()
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
            timeseries = masker.fit_transform(bold_file, confounds_file)
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

def estimate_connectivity(layout, subject, session, task, atlas, connectivity_metric='correlation', confounds=None, out_dir=None):

    """
    Makes connectivity matrices per subject per session per task per condition.
    Parameters
    ----------
    layout : str
        BIDS layout with derivatives indexed from pyBIDS
    subject : str
        Subject ID for which the networks will be calculated.
    session : str, optional
        Session of data collection. If there's only one session, we'll find it.
    connectivity_metric : {"correlation", "partial correlation", "tangent",\
                           "covariance", "precision"}, optional
        The matrix kind. Passed to Nilearn's `ConnectivityMeasure`.
    space : str
        'native' if analyses will be performed in subjects' functional native space (atlas(es) should be transformed)
        'mni152-2mm' if analyses will be performed in MNI125 2mm isotropic space (fMRI data should already be transformed)
    atlas : str
        Name of atlas for parcellating voxels into nodes, must be in the same `space` given above.
    confounds : list-like
        Names of confounds (should be columns in fmriprep output confounds.tsv).
    Returns
    -------
    adjacency_matrix
    """
    try:
        version = idconn.__version__
    except:
        version = 'test'
    if '.nii' in atlas:
        assert exists(atlas), f'Mask file does not exist at {atlas}'
    
    if not out_dir:
        deriv_dir = join(layout.root, 'derivatives', f'idconn-{version}')
    else:
        deriv_dir = out_dir
    atlas_name = basename(atlas).rsplit('.', 2)[0]
    # use pybids here to grab # of runs and preproc bold filenames
    connectivity_measure = connectome.ConnectivityMeasure(kind=connectivity_metric)
    bold_files = layout.get(scope='derivatives', return_type='file', suffix='bold', task=task, space='MNI152NLin2009cAsym',subject=subject, session=session, extension='nii.gz') # should be preprocessed BOLD file from fmriprep, grabbed with pybids
    print(f'BOLD files found at {bold_files}')
    confounds_files = layout.get(scope='derivatives', return_type='file', desc='confounds',subject=subject,session=session, task=task)

    runs = []
    if len(bold_files) > 1:
        for i in range(0, len(bold_files)):
            assert exists(bold_files[i]), "Preprocessed bold file(s) does not exist at {0}".format(bold_files)
            runs.append(layout.parse_file_entities(bold_files[i])['run'])
    else:
        runs = None
    print(f'Found runs: {runs}')

    out = join(deriv_dir,  f'sub-{subject}', f'ses-{session}', 'func')
    if not exists(out):
            makedirs(out)
    
    event_files = layout.get(return_type='filename', suffix='events', task=task, subject=subject)
    timing = pd.read_csv(event_files[0], header=0, index_col=0, sep='\t')
    conditions = timing['trial_type'].unique()

    if runs:
        corrmats = {}
        for run in runs:
            print('run = ', run)
            # read in events file for this subject, task, and run
            event_file = layout.get(return_type='filename', suffix='events', task=task, subject=subject, run=run, session=session)
            print('# of event files =', len(event_file), '\nfilename = ', event_file[0])
            the_file = str(event_file[0])
            assert exists(the_file), 'file really does not exist'
            timing = pd.read_csv(the_file, header=0, index_col=0, sep='\t')
            timing.sort_values('onset')

            confounds_file = layout.get(scope='derivatives', return_type='file', desc='confounds',subject=subject,session=session, task=task, run=run, extension='tsv')
            print(f'Confounds file located at: {confounds_file}')
            confounds_df = pd.read_csv(confounds_file[0], header=0, sep='\t')
            confounds_df = confounds_df[confounds].fillna(0)
            confounds_fname = join(deriv_dir,  f'sub-{subject}', f'ses-{session}', 'func', f'sub-{subject}_ses-{session}_task-{task}_run-{run}_desc-confounds_timeseries.tsv')
            confounds_df.to_csv(confounds_fname, sep='\t')

            bold_file = layout.get(scope='derivatives', return_type='file', suffix='bold', task=task, space='MNI152NLin2009cAsym',subject=subject, session=session, extension='nii.gz', run=run)
            assert len(bold_file) == 1, f'BOLD file improperly specified, more than one .nii.gz file with {subject}, {session}, {task}, {run}: {bold_file}'
            tr = layout.get_tr(bold_file)
            masker = input_data.NiftiLabelsMasker(atlas, standardize=True, t_r=tr, verbose=2)

            ex_bold = image.index_img(bold_file[0], 2)
            display = plotting.plot_epi(ex_bold)
            display.add_contours(atlas)
            display.savefig(join(deriv_dir,  f'sub-{subject}', f'ses-{session}', 'func', f'sub-{subject}_ses-{session}_task-{task}_run-{run}_desc-atlas_overlay.png'))
                
            print(f'BOLD file located at {bold_file}\nTR = {tr}s')
            try:
                #for each parcellation, extract BOLD timeseries
                print(f'Extracting bold signal for sub-{subject}, ses-{session}, run-{run}...')
                timeseries = masker.fit_transform(bold_file[0], confounds_fname)   
            except Exception as e:
                print('ERROR: Trying to extract BOLD signals, but', e)
            try:
            #and now we slice into conditions
                for condition in conditions:
                    blocks = []
                    cond_timing = timing[timing['trial_type'] == condition]
                    for i in cond_timing.index:
                        blocks.append((cond_timing.loc[i]['onset'] / tr, ((cond_timing.loc[i]['onset'] + cond_timing.loc[i]['duration']) / tr) + 1))
                    if len(blocks) > 1:
                        run_cond['{0}-{1}'.format(run, condition)] = np.vstack((timeseries[int(blocks[0][0]):int(blocks[0][1]), :], timeseries[int(blocks[1][0]):int(blocks[1][1]), :]))
                    if len(blocks) > 2:
                        for i in np.arange(2,len(blocks)):
                            run_cond['{0}-{1}'.format(run, condition)] = np.vstack((timeseries[int(blocks[0][0]):int(blocks[0][1]), :], timeseries[int(blocks[1][0]):int(blocks[1][1]), :]))
                        print('extracted signals for {0}, {1}, {2}'.format(task, run, condition), run_cond['{0}-{1}'.format(run, condition)].shape)
                    else:
                        pass
            except Exception as e:
                print('trying to slice and dice, but', e)
            try:
                print(f'Making correlation matrix for for sub-{subject}, ses-{session}, task-{task} ({condition}), run-{run}...')
                corrmats[run] = connectivity_measure.fit_transform([timeseries])[0]
            except Exception as e:
                print('ERROR: Trying to make corrmat, but', e)
        data = list(corrmats.values())
        stacked_corrmats = np.array(data)
        print('Stacked corrmats have dimensions', stacked_corrmats.shape)
        avg_corrmat = np.mean(stacked_corrmats, axis=0)
    else:
        confounds_file = layout.get(scope='derivatives', return_type='file', desc='confounds',subject=subject,session=session, task=task, extension='tsv')
        print(f'Confounds file located at: {confounds_file}')
        confounds_df = pd.read_csv(confounds_file[0], header=0, sep='\t')
        confounds_df = confounds_df[confounds].fillna(0)
        confounds_fname = join(deriv_dir,  f'sub-{subject}', f'ses-{session}', 'func', f'sub-{subject}_ses-{session}_task-{task}_desc-confounds_timeseries.tsv')
        confounds_df.to_csv(confounds_fname, sep='\t')

        bold_file = layout.get(scope='derivatives', return_type='file', suffix='bold', task=task, space='MNI152NLin2009cAsym',subject=subject, session=session, extension='nii.gz')
        assert len(bold_file) == 1, f'BOLD file improperly specified, more than one .nii.gz file with {subject}, {session}, {task}: {bold_file}'
        tr = layout.get_tr(bold_file)
        masker = input_data.NiftiLabelsMasker(atlas, standardize=True, t_r=tr, verbose=2)
	
        ex_bold = image.index_img(bold_file[0], 2)
        display = plotting.plot_epi(ex_bold)
        display.add_contours(atlas)
        display.savefig(join(deriv_dir,  f'sub-{subject}', f'ses-{session}', 'func', f'sub-{subject}_ses-{session}_task-{task}_desc-atlas_overlay.png'))
            
        print(f'BOLD file located at {bold_file}\nTR = {tr}s')
        try:
            #for each parcellation, extract BOLD timeseries
            print(f'Extracting bold signal for sub-{subject}, ses-{session}...')
            timeseries = masker.fit_transform(bold_file[0], confounds_fname)   
        except Exception as e:
            print('ERROR: Trying to extract BOLD signals, but', e)
        try:
            print(f'Making correlation matrix for for sub-{subject}, ses-{session}...')
            avg_corrmat = connectivity_measure.fit_transform([timeseries])[0]
        except Exception as e:
            print('ERROR: Trying to make corrmat, but', e)

    print('Correlation matrix created, dimensions:', avg_corrmat.shape)
    try:
        corrmat_df = pd.DataFrame(index=np.arange(1, avg_corrmat.shape[0]+1), columns=np.arange(1, avg_corrmat.shape[0]+1),data=avg_corrmat)
        corrmat_file = join(deriv_dir,  
                            f'sub-{subject}', 
                            f'ses-{session}', 
                            'func', 
                            f'sub-{subject}_ses-{session}_task-{task}_space-MNI152NLin2009cAsym_atlas-{atlas_name}_desc-corrmat_bold.tsv')
        corrmat_df.to_csv(corrmat_file, sep='\t')
    except Exception as e:
        print('ERROR saving corrmat...', e)
    return corrmat_df, corrmat_file
