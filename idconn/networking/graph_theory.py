import numpy as np
#import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join
#from nilearn.connectome import ConnectivityMeasure
from scipy.sparse.csgraph import minimum_spanning_tree
import bct
#import datetime

def omst(matrix, density=True, plot=False):
    '''
    WARNING: THIS IS SLOW AF, REPLACING WITH NETWORKX VERSION IN NEAR FUTURE
    '''
    dims = matrix.shape
    if matrix.ndim > 2:
        raise ValueError("'matrix' should be a 2D array. "
                         "An array with %d dimension%s was passed"
                         % (matrix.ndim,
                            "s" if matrix.ndim > 1 else ""))
    else:
        mst = minimum_spanning_tree(matrix)
        mst_arr = mst.toarray().astype(float)
        matrix_2 = np.where(mst_arr != 0, 0, matrix)
        cost = np.sum(matrix_2) / np.sum(matrix)
        Eg = bct.efficiency_wei(matrix_2)
        trees = [mst_arr]
        GCE = [Eg - cost]
        Cost = [cost]

        while np.sum(matrix_2) > 1000:
            #print(np.sum(matrix_2))
            mst = minimum_spanning_tree(matrix_2)
            mst_arr = mst.toarray().astype(float)
            matrix_2 = np.where(mst_arr != 0, 0, matrix_2)
            cost = np.sum(matrix_2) / np.sum(matrix)
            Eg = bct.efficiency_wei(matrix_2)
            trees.append(mst_arr)
            GCE.append(Eg - cost)
            Cost.append(cost)
        trees = np.asarray(trees)
        max_value = max(GCE)
        max_GCE = GCE.index(max_value)
        thresholded = np.sum(trees[:max_GCE, :, :], axis=0)
        if plot == True:
            fig,ax = plt.subplots()
            sns.lineplot(Cost, GCE, ax=ax, palette='husl')
            plt.scatter(Cost[max_GCE], 
                        GCE[max_GCE], 
                        marker='x', 
                        edgecolors=None, 
                        c='magenta')
            ax.set_ylabel('Global Cost Efficiency')
            ax.set_xlabel('Cost')
            
        if density == True:
            den = np.sum(thresholded != 0) / (dims[0] * dims[1])
            return thresholded, den
    return thresholded, fig

def graph_auc(matrix, thresholds, measure, args):
    '''
    matrix : array
    measure : function from bctpy
    '''
    from bct import measure, threshold_proportional

    metrics = []
    for p in np.arange(thresholds[0], thresholds[1], 0.01):
        thresh = threshold_proportional(matrix, p, copy=True)
        metric = measure(thresh, args)
        metrics.append(metric)
    auc= np.trapz(metrics, dx=0.01)
    return auc

def graph_omst(matrix, measure, args):
    from bct import measure
    # threshold using orthogonal minimum spanning tree
    thresh_mat = omst(matrix)

    # calculate graph measure on thresholded matrix
    metric = measure(thresh_mat, args)
    return metric