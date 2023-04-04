import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from os.path import join

# from nilearn.connectome import ConnectivityMeasure
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import skew
import bct

# import datetime


def avg_corrmat(ppt_df):
    """
    Reads in adjacency matrices from the pandas df with ppt info and adj, then computes an average.
    """
    stacked_corrmats = np.array(ppt_df["adj"])
    print("Stacked corrmats have dimensions", stacked_corrmats.shape)
    avg_corrmat = np.mean(stacked_corrmats, axis=0)
    return avg_corrmat


def null_model(W, bin_swaps=5, wei_freq=0.1, seed=None):
    def get_rng(seed):
        if seed is None or seed == np.random:
            return np.random.mtrand._rand
        elif isinstance(seed, np.random.RandomState):
            return seed
        try:
            rstate = np.random.RandomState(seed)
        except ValueError:
            rstate = np.random.RandomState(np.random.Random(seed).randint(0, 2**32 - 1))
        return rstate

    def randmio_und_signed(R, itr, seed=None):
        rng = get_rng(seed)
        R = R.copy()
        n = len(R)

        itr *= int(n * (n - 1) / 2)

        max_attempts = int(np.round(n / 2))
        eff = 0

        for it in range(int(itr)):
            att = 0
            while att <= max_attempts:
                a, b, c, d = pick_four_unique_nodes_quickly(n, rng)

                r0_ab = R[a, b]
                r0_cd = R[c, d]
                r0_ad = R[a, d]
                r0_cb = R[c, b]

                # rewiring condition
                if (
                    np.sign(r0_ab) == np.sign(r0_cd)
                    and np.sign(r0_ad) == np.sign(r0_cb)
                    and np.sign(r0_ab) != np.sign(r0_ad)
                ):
                    R[a, d] = R[d, a] = r0_ab
                    R[a, b] = R[b, a] = r0_ad

                    R[c, b] = R[b, c] = r0_cd
                    R[c, d] = R[d, c] = r0_cb

                    eff += 1
                    break

                att += 1

        return R, eff

    def pick_four_unique_nodes_quickly(n, seed=None):
        """
        This is equivalent to np.random.choice(n, 4, replace=False)
        Another fellow suggested np.random.random_sample(n).argpartition(4) which is
        clever but still substantially slower.
        """
        rng = get_rng(seed)
        k = rng.randint(n**4)
        a = k % n
        b = k // n % n
        c = k // n**2 % n
        d = k // n**3 % n
        if a != b and a != c and a != d and b != c and b != d and c != d:
            return (a, b, c, d)
        else:
            # the probability of finding a wrong configuration is extremely low
            # unless for extremely small n. if n is extremely small the
            # computational demand is not a problem.

            # In my profiling it only took 0.4 seconds to include the uniqueness
            # check in 1 million runs of this function so I think it is OK.
            return pick_four_unique_nodes_quickly(n, rng)

    rng = get_rng(seed)
    if not np.allclose(W, W.T):
        print("Input must be undirected")
    W = W.copy()
    n = len(W)
    np.fill_diagonal(W, 0)  # clear diagonal
    Ap = W > 0  # positive adjmat
    An = W < 0  # negative adjmat

    if np.size(np.where(Ap.flat)) < (n * (n - 1)):
        W_r, eff = randmio_und_signed(W, bin_swaps, seed=rng)
        Ap_r = W_r > 0
        An_r = W_r < 0
    else:
        Ap_r = Ap
        An_r = An

    W0 = np.zeros((n, n))
    for s in (1, -1):
        if s == 1:
            Acur = Ap
            A_rcur = Ap_r
        else:
            Acur = An
            A_rcur = An_r

        S = np.sum(W * Acur, axis=0)  # strengths
        Wv = np.sort(W[np.where(np.triu(Acur))])  # sorted weights vector
        i, j = np.where(np.triu(A_rcur))
        (Lij,) = np.where(np.triu(A_rcur).flat)  # weights indices

        P = np.outer(S, S)

        if wei_freq == 0:  # get indices of Lij that sort P
            Oind = np.argsort(P.flat[Lij])  # assign corresponding sorted
            W0.flat[Lij[Oind]] = s * Wv  # weight at this index
        else:
            wsize = np.size(Wv)
            wei_period = np.round(1 / wei_freq).astype(int)  # convert frequency to period
            lq = np.arange(wsize, 0, -wei_period, dtype=int)
            for m in lq:  # iteratively explore at this period
                # get indices of Lij that sort P
                Oind = np.argsort(P.flat[Lij])
                R = rng.permutation(m)[: np.min((m, wei_period))]
                for q, r in enumerate(R):
                    # choose random index of sorted expected weight
                    o = Oind[r]
                    W0.flat[Lij[o]] = s * Wv[r]  # assign corresponding weight

                    # readjust expected weighted probability for i[o],j[o]
                    f = 1 - Wv[r] / S[i[o]]
                    P[i[o], :] *= f
                    P[:, i[o]] *= f
                    f = 1 - Wv[r] / S[j[o]]
                    P[j[o], :] *= f
                    P[:, j[o]] *= f

                    # readjust strength of i[o]
                    S[i[o]] -= Wv[r]
                    # readjust strength of j[o]
                    S[j[o]] -= Wv[r]

                O = Oind[R]
                # remove current indices from further consideration
                Lij = np.delete(Lij, O)
                i = np.delete(i, O)
                j = np.delete(j, O)
                Wv = np.delete(Wv, R)

    W0 = W0 + W0.T
    return W0


def generate_null(ppt_df, thresh_arr, measure, permutations=1000):
    """
    Generate a distribution of graph measure values based on a null connectivity matrix
    that is like the average connectivity matrix across participants.

    """
    null_dist = pd.DataFrame(index=range(0, permutations), columns=["mean", "sdev"])
    avg_corr = avg_corrmat(ppt_df)
    eff_perm = []
    j = 0
    while j < permutations:
        effs = []
        W = null_model(avg_corr.values)
        for thresh in thresh_arr:
            thresh_corr = bct.threshold_proportional(W, thresh)
            leff = measure(thresh_corr)
            effs.append(leff)
        effs_arr = np.asarray(effs)
        leff_auc = np.trapz(effs_arr, dx=0.03, axis=0)
        eff_perm.append(leff_auc)
        j += 1

    return null_dist


def omst(matrix, density=True, plot=False):
    """
    WARNING: THIS IS SLOW AF, REPLACING WITH NETWORKX VERSION IN NEAR FUTURE
    """
    dims = matrix.shape
    if matrix.ndim > 2:
        raise ValueError(
            "'matrix' should be a 2D array. "
            "An array with %d dimension%s was passed"
            % (matrix.ndim, "s" if matrix.ndim > 1 else "")
        )
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
            # print(np.sum(matrix_2))
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
            fig, ax = plt.subplots()
            sns.lineplot(Cost, GCE, ax=ax, palette="husl")
            plt.scatter(Cost[max_GCE], GCE[max_GCE], marker="x", edgecolors=None, c="magenta")
            ax.set_ylabel("Global Cost Efficiency")
            ax.set_xlabel("Cost")

        if density == True:
            den = np.sum(thresholded != 0) / (dims[0] * dims[1])
            return thresholded, den
    return thresholded, fig


def graph_auc(matrix, thresholds, measure, args):
    """
    matrix : array
    measure : function from bctpy
    """
    from bct import measure, threshold_proportional

    metrics = []
    for p in np.arange(thresholds[0], thresholds[1], 0.01):
        thresh = threshold_proportional(matrix, p, copy=True)
        metric = measure(thresh, args)
        metrics.append(metric)
    auc = np.trapz(metrics, dx=0.01)
    return auc


def graph_omst(matrix, measure, args):
    from bct import measure

    # threshold using orthogonal minimum spanning tree
    thresh_mat = omst(matrix)

    # calculate graph measure on thresholded matrix
    metric = measure(thresh_mat, args)
    return metric


def scale_free_tau(corrmat, skew_thresh, proportional=True):
    """'
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
    """
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
    """
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
    """
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
