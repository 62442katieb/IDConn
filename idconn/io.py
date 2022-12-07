import bids
import json
from nilearn import datasets
import nibabel as nib
from os.path import exists, join, basename


import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
#from matplotlib import projections
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from nilearn import datasets, plotting, surface


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
    #nilearn_coord = ['power_2011', 'dosenbach_2010', 'seitzman_2018']
    #nilearn_4d = ['allen_2011', '']
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

    return atlas, path

def vectorize_corrmats(matrices):
    """Returns the vectorized upper triangles of a 3-dimensional array
    (i.e., node x node x matrix) of matrices. Output will be a 2-dimensional
    array (i.e., matrix x node^2)
    Parameters
    ----------
    matrices : numpy array of shape (p, n, n)
        Represents the link strengths of the graphs. Assumed to be
        an array of symmetric nxn matrices per participant and/or timepoint (p).
    
    Returns
    -------
    edge_vector : numpy array of shape (p, n^2)
        Represents an array of vectorized upper triangles of 
        the input matrices.
    """
    #print(f'\n\n\n{matrices.shape}, {matrices.ndim}\n\n\n')
    num_node = matrices.shape[1]
    upper_tri = np.triu_indices(num_node, k=1)
    if matrices.ndim == 3:
        num_node = matrices.shape[1]
        upper_tri = np.triu_indices(num_node, k=1)
        num_matrices = matrices.shape[0]
        edge_vector = []
        for matrix in range(0,num_matrices):
            vectorized = matrices[matrix,:,:][upper_tri]
            edge_vector.append(vectorized)
    
    elif matrices.ndim == 2:
        true = matrices[0].T == matrices[0]
        if true.all():
            edge_vector = matrices[upper_tri]
        else:
            print('Matrices of incompatible shape:', matrices.shape, 
                '\nNumber of dimensions needs to be 3 (node x node x participant) or 2 (node x node).')
    elif matrices.ndim == 1:
        if matrices[0].ndim == 2:
            num_node = matrices[0].shape[0]
            upper_tri = np.triu_indices(num_node, k=1)
            edge_vector = []
            for matrix in matrices:
                vectorized = matrix[upper_tri]
                edge_vector.append(vectorized)
        else:
            print('Matrices of incompatible shape:', matrices.shape, 
                  '\nNumber of dimensions needs to be 3 (node x node x participant) or 2 (node x node).')
    edge_vector = np.asarray(edge_vector)
    return edge_vector

def read_corrmats(layout, task, deriv_name='IDConn', atlas=None, conf_measures=None, z_score=True, vectorized=True, verbose=False):
    """Returns a node x node x (subject x session) matrix of correlation matrices  
    from a BIDS derivative folder. Optionally returns a subject x session dataframe
    of confound measures (e.g., motion averages) and/or a node^2 x (subject x session) 
    array of vectorized upper triangles of those correlation matrices.
    Parameters
    ----------
    matrices : numpy array of shape (n, n, p)
        Represents the link strengths of the graphs. Assumed to be
        an array of symmetric matrices.
    
    Returns
    -------
    edge_vector : numpy array of shape (p, n^2)
        Represents an array of vectorized upper triangles of 
        the input matrices.
    """
    subjects = layout.get(return_type='id', 
                          target='subject', 
                          suffix='bold', 
                          scope=deriv_name
                         )
    all_sesh = layout.get(return_type='id',
           target='session',
           task=task, 
           suffix='bold',
           scope=deriv_name
          )
    ppts_fname = layout.get_file('participants.tsv').path
    ppt_df = pd.read_csv(ppts_fname, sep='\t', index_col=[0,1])
    ppt_df['adj'] = ''
    if vectorized:
        ppt_df['edge_vector'] = ''
    
    for subject in subjects:
        if verbose:
            print(subject)
        else:
            pass
        sessions = layout.get(return_type='id', 
                              target='session', 
                              task=task, 
                              suffix='bold', 
                              subject=subject, 
                              scope=deriv_name)
        
        for session in sessions:
            if verbose:
                print(session)
            else:
                pass
            path = layout.get(return_type='filename',
                               task=task, 
                               subject=subject,
                               session=session,
                               suffix='bold',
                               scope='IDConn', 
                               atlas=atlas,
                              )
            if verbose:
                print(f'Corrmat path for sub-{subject}, ses-{session}: \t{path}')
            else:
                pass
            if type(path) == list:
                #print(path)
                path = path[0]
            else:
                pass
            assert exists(path), f'Corrmat file not found at {path}'
            adj_matrix = pd.read_csv(path, sep='\t', header=0, index_col=0)
            if z_score == True:
                z_adj = np.arctanh(adj_matrix.values)
                z_adj = np.where(z_adj == np.inf, 0, z_adj)
                #print(z_adj.shape)
                ppt_df.at[(f'sub-{subject}', 
                           f'ses-{session}'), 
                          'adj'] = z_adj
            else:
                #print(adj_matrix.values.shape)
                ppt_df.at[(f'sub-{subject}', 
                           f'ses-{session}'), 
                          'adj'] = adj_matrix.values
                
            
            if vectorized == True:
                edge_vector = vectorize_corrmats(adj_matrix.values)
                #print(edge_vector.shape)
                ppt_df.at[(f'sub-{subject}', 
                                   f'ses-{session}'), 
                                  'edge_vector'] = edge_vector
    ppt_df.replace({'': np.nan}, inplace=True)
    return ppt_df

def undo_vectorize(edges, num_node):
    #j = len(edges)
    #num_node = (np.sqrt((8 * j) + 1) + 1) / 2
    X = np.zeros((num_node,num_node))
    X[np.triu_indices(X.shape[0], k = 1)] = edges
    X = X + X.T
    return X

def plot_edges(adj, atlas_nii, threshold=None, title=None, strength=False, cmap='vlag', node_size='strength'):
    coords = plotting.find_parcellation_cut_coords(atlas_nii)
    num_node = adj.shape[0]
    # only plot the top t% of edges
    if threshold == 'computed':
        threshold = f'{(1 - (100 / num_node ** 2)) * 100}%'
    elif type(threshold) == float or type(threshold) == int:
        threshold = f'{threshold}%'
    else:
        threshold = '99%'
    print('edge plotting threshold: ', threshold)

    if node_size == 'strength':
        node_strength = np.sum((np.abs(adj)), axis=0)
        node_strength /= np.max(node_strength)
        node_strength **= 4
        node_size = node_strength
    fig = plt.figure(figsize=(12,4))
    if title is not None:
        fig.suptitle(title)
    gs = GridSpec(1, 2, width_ratios=[4,2])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    plt.tight_layout(w_pad=5)
    g = plotting.plot_connectome(adj, coords, 
                                node_size=node_size,
                                edge_threshold=threshold, 
                                edge_cmap=cmap, 
                                figure=fig, 
                                axes=ax0,
                                colorbar=False, 
                                annotate=False)
    h = sns.heatmap(adj, square=True, cmap=cmap, ax=ax1, center=0)
    if strength:
        fig2 = plt.figure(figsize=(12,4))
        if title is not None:
            fig2.suptitle(title)
        fsaverage = datasets.fetch_surf_fsaverage()
        nimg = nib.load(atlas_nii)
        regn_sch_arr = nimg.get_fdata()
        for i in np.arange(0,num_node):
            regn_sch_arr[np.where(regn_sch_arr == i+1)] = np.sum(adj[i])
        strength_nimg = nib.Nifti1Image(regn_sch_arr, nimg.affine)
        nib.save(strength_nimg, '/Users/katherine.b/Dropbox/HC_Use_predictive-strength.nii')

        gs = GridSpec(1, 4)
        # plot edge weights on surfaces
        ax2 = fig2.add_subplot(gs[0], projection='3d')
        ax3 = fig2.add_subplot(gs[1], projection='3d')
        ax4 = fig2.add_subplot(gs[2], projection='3d')
        ax5 = fig2.add_subplot(gs[3], projection='3d')

        texture_l = surface.vol_to_surf(strength_nimg, fsaverage.pial_left, interpolation='nearest')
        texture_r = surface.vol_to_surf(strength_nimg, fsaverage.pial_right, interpolation='nearest')

        plt.tight_layout(w_pad=-1)
        i = plotting.plot_surf_stat_map(fsaverage.pial_left, texture_l, symmetric_cbar=False, threshold=0.5,
                                                cmap=cmap, view='lateral', colorbar=False, axes=ax2)
        j = plotting.plot_surf_stat_map(fsaverage.pial_left, texture_l, symmetric_cbar=False, threshold=0.5,
                                                cmap=cmap, view='medial', colorbar=False, axes=ax3)
        k = plotting.plot_surf_stat_map(fsaverage.pial_right, texture_r, symmetric_cbar=False, threshold=0.5,
                                                cmap=cmap, view='lateral', colorbar=False, axes=ax4)
        l = plotting.plot_surf_stat_map(fsaverage.pial_right, texture_r, symmetric_cbar=False, threshold=0.5,
                                                cmap=cmap, view='medial', colorbar=False, axes=ax5)
        return fig, fig2
    else:
        return fig