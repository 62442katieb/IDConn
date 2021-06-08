import bids
import json
from nilearn import datasets

def build_outdir(dset_dir, subjects, sessions, version):
    try:
        if not exists(join(dset_dir, 'derivatives/idconn-{0}'.format(version), subject, session)):
            makedirs(join(dset_dir, 'derivatives/idconn-{0}'.format(version), subject, session))
    except Exception as e:
	    print('Problem making output directories:', e)

def build_statsmodel_json(name, task, contrast, confounds, highpass, 
                          mask, conn_meas, graph_meas=None, exclude=None, outfile=None):
    '''
    Creates a BIDS Stats Models json with analysis details for further use.

    Parameters
    ----------
    root_dir : str
        Location of BIDS dataset root
    validate : bool
        If true, pybids will check if this is a valid BIDS-format
        dataset before continuing.
    absolute_paths : bool
        If true, will assume paths are absolute, instead of relative.
    derivatives : str
        Location of preprocessed data (i.e., name of fmriprep dir).
    verbose : bool
        If true, will narrate finding of dataset and describe it.
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
    
    '''
    mask_builtins = ['shen270', 'craddock270', 'schaefer400', 'yeo7', 'yeo17']
    if '.nii' in mask:
        assert exists(mask), 'Mask file does not exist at {mask}'.format(mask=mask)
        if '.gz' in mask:
            mask_name = basename(mask).rsplit('.', 2)[0]
        else:
            mask_name = basename(mask).rsplit('.', 1)[0]
    else:
        assert mask in mask_builtins, 'Mask {mask} not in built-in mask options. Please provide file path or one of {mask_builtins}'.format(mask=mask, mask_builtins=mask_builtins)
    variables = confounds + ["{mask_name}*".format(mask_name=mask_name)]
    statsmodel = {
        "name": name,
        "description": "A functional connectivity analysis of {task}, comparing {contrast}".format(task=task, 
                                                                                                   contrast=contrast), 
        "input":{
            "task": task
        },
        "blocks":[{
                "level": "run",
                "transformations":{
                        "name": "load_image_data",
                        "input": ["bold"],
                        "aggregate": ["mean"],
                        "mask": [mask_name],
                        "output": ["{mask_name}*".format(mask_name=mask_name)]
                    },
        },
            {
                "level": "session",
                "model": {
                    "variables": variables,
                    "options": {
                        "confounds": confounds,
                        "high_pass_filter_cutoff_secs": highpass
                    },
                    "variances": {
                        "name": "session_level",
                        "groupBy": "session"
                    },
                    "software": {
                        "IDConn": {
                            "ConnectivityMeasure": [conn_meas],
                            "GraphMetrics": [graph_meas]
                        }
                    }
                }
                
            }
        ]
    }
    statsmodel_json = json.dumps(statsmodel, indent = 2)
    
    outfile = '{name}-statsmodel.json'.format(name=name)
    with open(outfile, 'w') as outfile:
        json.dump(statsmodel, outfile)
    return statsmodel_json


def bids_io(root_dir, validate=True, absolute_paths=True, derivatives='fmriprep', task= , subjects=, sessions=None, verbose=False):
    """
    Wraps pybids to provide relevant input for IDConn and output paths for results.
    Not implemented yet.
    Parameters
    ----------
    root_dir : str
        Location of BIDS dataset root
    validate : bool
        If true, pybids will check if this is a valid BIDS-format
        dataset before continuing.
    absolute_paths : bool
        If true, will assume paths are absolute, instead of relative.
    derivatives : str
        Location of preprocessed data (i.e., name of fmriprep dir).
    verbose : bool
        If true, will narrate finding of dataset and describe it.
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
    layout = bids.BIDSLayout(root_dir, validate=validate, absolute_paths=absolute_paths)
    subj_with_fmri = layout.get(return_type='id', target='subject', suffix='func')

    print('There are', len(subj_with_fmri), 'subjects with fMRI data:', subj_with_fmri)

    return fmriprep_dir, idconn_dir

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