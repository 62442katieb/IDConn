import numpy as np
import networkx as nx
import pandas as pd
import bct 


def scale_free_tau(corrmat, skew_thresh, proportional=True):
    ''''
    Calculates threshold at which network becomes scale-free, estimated from the skewness of the networks degree distribution.
    Parameters
    ----------
    corrmat : numpy.array
        Correlation or other connectivity matrix from which tau_connected will be estimated.
        Should be values between 0 and 1.
    proportional : bool
        Determines whether connectivity matrix is thresholded proportionally or absolutely.
        Default is proportional as maintaining network density across participants is a priority
    Returns
    -------
    tau : float
        Lowest vaue of tau (threshold) at which network is scale-free.
    '''
    tau = 0.01
    skewness = 1
    while abs(skewness) > 0.3:
        if proportional:
            w = bct.threshold_proportional(corrmat, tau)
        else:
            w = bct.threshold_absolute(corrmat, tau)
        skewness = skew(bct.degrees_und(w))
        tau += 0.01
    return tau

def connected_tau(corrmat, proportional=True):
    '''
    Calculates threshold at network becomes node connected, using NetworkX's `is_connected` function.
    Parameters
    ----------
    corrmat : numpy.array
        Correlation or other connectivity matrix from which tau_connected will be estimated.
        Should be values between 0 and 1.
    proportional : bool
        Determines whether connectivity matrix is thresholded proportionally or absolutely.
        Default is proportional as maintaining network density across participants is a priority
    Returns
    -------
    tau : float
        Highest vaue of tau (threshold) at which network becomes node-connected.
    '''
    tau = 0.01
    connected = False
    while connected == False:
        if proportional:
            w = bct.threshold_proportional(corrmat, tau)
        else:
            w = bct.threshold_absolute(corrmat, tau)
        w_nx = nx.convert_matrix.from_numpy_array(w)
        connected = nx.algorithms.components.is_connected(w_nx)
        tau += 0.01
    return tau