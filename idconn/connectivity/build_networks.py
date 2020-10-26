import numpy as np
import pandas as pd
import nibabel as nib
from os import makedirs
from os.path import join, exists
from glob import glob
from nilearn import input_data, datasets, connectome

#from .utils import contrast

def atlas_picker(atlas, path, key=None):
    """Takes in atlas name and path to file, if local, returns
    nifti-like object (usually file path to downloaded atlas),
    and atlas name (for tagging output files). If atlas is from
    Nilearn, will download atlas, **and space must be == 'MNI'.
    If atlas is provided by user (path must be specified), then
    space of atlas must match space of fMRI data, but that is up
    to the user to determine.
    Parameters
    ----------
    atlas : str
        Name of the atlas/parcellation used to define nodes from 
        voxels. If using an atlas fetchable by Nilearn, atlas name 
        must match the function `fetch_atlas_[name]`.
    path : str
        Path to the atlas specified, if not using a dataset from Nilearn. 
        If using `nilearn.datasets` to fetch an atlas, will revert to 
        `derivatives/idconn` path.
    key : str
        Atlas-specific key for denoting which of multiple versions
        will be used. Default behavior is described in the "atlases"
        section of the docs. NOT IMPLEMENTED
    Returns
    -------
    atlas : str
        Name of the atlas chosen.
    path : str
        File path of atlas. If user-provided, will be copied into
        `derivatives/idconn`. If using an atlas from Nilearn, will
        be path to downloaded nifti.
    shape : str
        Indicates shape of map (3d, 4d, coords) for choosing appropriate
        Nilearn masker for extracting BOLD signals from nifti files.
    """
    nilearn_3d = ['craddock_2012', 'destrieux_2009', 'harvard_oxford', 'smith_2009', 'yeo_2011', 'aal', 'pauli_2017', 'msdl']
    nilearn_coord = ['power_2011', 'dosenbach_2010', 'seitzman_2018']
    nilearn_4d = ['allen_2011', '']
    if atlas in nilearn_3d:
        if atlas == 'craddock_2012':
            atlas_dict = datasets.fetch_atlas_craddock_2012(data_dir=path)
            atlas_path = atlas_dict['tcorr_2level']
            nifti = nib.load(atlas_path)
            nifti_arr = nifti.get_fdata()
            #selecting one volume of the nifti, each represent different granularity of parcellation
            #selecting N = 270, the 27th volume per http://ccraddock.github.io/cluster_roi/atlases.html
            nifti = nib.Nifti1Image(nifti_arr[:,:,:,26], nifti.affine)
            nifti.to_filename()

    return atlas, path, dimension

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

def bids_io(root_dir, validate=True, absolute_paths=True, derivatives='fmriprep'):
    """
    Wraps pybids to provide relevant input for IDConn and output paths for results.
    Not implemented yet.
    """
    pass
    

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

def rest_networks(dset_dir, subject, session, runs, connectivity_metric, space, atlas, confounds):
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
    connectivity_metric : {"correlation", "partial correlation", "tangent",\
                           "covariance", "precision"}, optional
        The matrix kind. Passed to Nilearn's `ConnectivityMeasure`.
    space : str
        'native' if analyses will be performed in subjects' functional native space (atlas(es) should be transformed)
        'mni152-2mm' if analyses will be performed in MNI125 2mm isotropic space (fMRI data should already be transformed)
    atlas : str
        Name of atlas for parcellating voxels into nodes, passed to `atlas_picker`
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
    if not 'atlas' in atlas:
        atlas = 'atlas-{0}'.format(atlas)
    subj_dir = join(dset_dir, subject, session, 'func')
    preproc_dir = join(dset_dir, 'derivatives/idconn/preproc', subject, session, 'func')
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
        bold_file = join(preproc_dir,
                        '{0}_{1}_{2}_{3}_bold-mcf.nii.gz'.format(subject, session, task, run))
        assert exists(bold_file), "epi_cleaned does not exist at {0}".format(bold_file)
    
        if space == 'native':
            atlas_file = join(preproc_dir,
                              '{0}_{1}_{2}_{3}_{4}.nii.gz'.format(subject, session, task, run, atlas))
            assert exists(atlas_file), 'atlas/parcellation not found at {0}'.format(atlas)
        elif 'mni' in space:
            atlas, atlas_file, dim = atlas_picker(atlas, path=join(dset_dir, 'derivatives/idconn/{0}'.format(version)), key=key)
            #add in option here to use nilearn-grabbed atlases
        elif 'MNI' in space:
            atlas, atlas_file, dim = atlas_picker(atlas, path=join(dset_dir, 'derivatives/idconn/{0}'.format(version)), key=key)
        else:
            raise NameError('You specified an unknown image space! Space should be some version of MNI or subject native space, and you provided {0}'.format(space))
        #LATER: PRINT OVERLAY OF MASK ON EXAMPLE FUNC
        if len(confounds) > 1:
            if not exists(join(preproc_dir,
                               '{0}_{1}_{2}_{3}_bold-confounds+outliers.tsv'.format(subject, session, task, run))):
                confounds_file = confounds_merger(confounds)
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