from posixpath import sep
import numpy as np
import pandas as pd
import idconn.connectivity.build_networks
from os import makedirs
from os.path import join, exists, basename
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
    

def task_connectivity(layout, subject, session, task, atlas, confounds, connectivity_metric='correlation', out_dir=None):
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
    confounds : list-like
        Filenames of confounds files.
    Returns
    -------
    confounds_file : str
        Filename of merged confounds .tsv file
    """
    #version = '0.1.1'
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
    space = 'MNI152NLin2009cAsym'
    atlas_name = basename(atlas).rsplit('.', 2)[0]
    # use pybids here to grab # of runs and preproc bold filenames
    connectivity_measure = connectome.ConnectivityMeasure(kind=connectivity_metric)
    bold_files = layout.get(scope='derivatives', return_type='file', suffix='bold', task=task, space=space,subject=subject, session=session, extension='nii.gz') # should be preprocessed BOLD file from fmriprep, grabbed with pybids
    print(f'BOLD files found at {bold_files}')

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

    run_cond = {}
    corrmats = {}
    for run in runs:
        bold_file = layout.get(scope='derivatives', return_type='file', suffix='bold', task=task, space='MNI152NLin2009cAsym',subject=subject, session=session, extension='nii.gz', run=run)
        assert len(bold_file) == 1, f'BOLD file improperly specified, more than one .nii.gz file with {subject}, {session}, {task}, {run}: {bold_file}'
        tr = layout.get_tr(bold_file)
	
        #load timing file 
        #update to use pyBIDS + layout
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

        masker = input_data.NiftiLabelsMasker(atlas, standardize=True, t_r=tr, verbose=2)
        ex_bold = image.index_img(bold_file[0], 2)
        display = plotting.plot_epi(ex_bold)
        display.add_contours(atlas)
        display.savefig(join(deriv_dir,  f'sub-{subject}', f'ses-{session}', 'func', f'sub-{subject}_ses-{session}_task-{task}_run-{run}_space-MNI152NLin2009cAsym_space-{atlas_name}_overlay.png'))
            
        print(f'BOLD file located at {bold_file}\nTR = {tr}s')
        
        masker = input_data.NiftiLabelsMasker(atlas, standardize=True, t_r=tr, verbose=1)
        timeseries = masker.fit_transform(bold_file[0], confounds=confounds_fname)
        #load timing file 
        #update to use pyBIDS + layout
        try:
            #and now we slice into conditions
            for condition in conditions:
                run_cond[condition] = {}
                corrmats[condition] = {}
                blocks = []
                cond_timing = timing[timing['trial_type'] == condition]
                for i in cond_timing.index:
                    blocks.append((cond_timing.loc[i]['onset'] / tr, ((cond_timing.loc[i]['onset'] + cond_timing.loc[i]['duration']) / tr) + 1))
                if len(blocks) > 1:
                    run_cond[condition][run] = np.vstack((timeseries[int(blocks[0][0]):int(blocks[0][1]), :], timeseries[int(blocks[1][0]):int(blocks[1][1]), :]))
                if len(blocks) > 2:
                    for i in np.arange(2,len(blocks)):
                        run_cond[condition][run] = np.vstack((timeseries[int(blocks[0][0]):int(blocks[0][1]), :], timeseries[int(blocks[1][0]):int(blocks[1][1]), :]))
                    #print('extracted signals for {0}, {1}, {2}'.format(task, run, condition), run_cond['{0}-{1}'.format(run, condition)].shape)
                else:
                    pass
                print(f'Making correlation matrix for {run}, {condition}.')
                corrmats[condition][run] = connectivity_measure.fit_transform([run_cond[condition][run]])[0]
                print('And that correlation matrix is', corrmats[condition][run].shape)
        except Exception as e:
            print('trying to slice and dice, but', e)
    #and paste together the timeseries from each run together per condition
    files = []
    avg_corrmats = {}
    print('Corrmats per run per condition have been made!')
    for condition in conditions:
        print(f'Merging corrmats for {task}-{condition}...')
        data = list(corrmats[condition].values())
        stacked_corrmats = np.array(data)
        print('Stacked corrmats have dimensions', stacked_corrmats.shape)
        avg_corrmat = np.mean(stacked_corrmats, axis=0)
        corrmat_df = pd.DataFrame(index=np.arange(1, avg_corrmat.shape[0]+1), columns=np.arange(1, avg_corrmat.shape[0]+1),data=avg_corrmat)
        avg_corrmats[condition] = corrmat_df
        corrmat_file = join(deriv_dir,  
                            f'sub-{subject}', f'ses-{session}', 'func', f'sub-{subject}_ses-{session}_task-{task}_desc-{condition}_space-MNI152NLin2009cAsym_atlas-{atlas_name}_corrmat.tsv')
        try:
            corrmat_df.to_csv(corrmat_file, sep='\t')
            files.append(corrmat_file)
        except Exception as e:
            print('saving corrmat...', e)
    return files, avg_corrmats

def connectivity(layout, subject, session, task, atlas, connectivity_metric='correlation', confounds=None, out_dir=None):

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
    
    
    #event_files = layout.get(return_type='filename', suffix='events', task=task, subject=subject)
    #timing = pd.read_csv(event_files[0], header=0, index_col=0, sep='\t')
    #conditions = timing['trial_type'].unique()

    if runs:
        corrmats = {}
        for run in runs:
            print('run = ', run)
            # read in events file for this subject, task, and run
            

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
                print(f'Making correlation matrix for for sub-{subject}, ses-{session}, task-{task}, run-{run}...')
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
