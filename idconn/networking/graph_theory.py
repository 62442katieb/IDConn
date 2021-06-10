import numpy as np
import pandas as pd
from os.path import join
from nilearn.connectome import ConnectivityMeasure
import bct
import datetime


def compute(matrix, thresholds, measure, args):
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