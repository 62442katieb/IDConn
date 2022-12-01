import numpy as np
import statsmodels as sm
import networkx as nx
from utils import vectorize_corrmats, undo_vectorize


def pynbs(matrices, outcome, confounds, alpha, predict=False, permutations=10000, stratified=False):
    '''
    Calculates the Network Based Statistic (Zalesky et al., 2011) on connectivity matrices provided
    of shape ((subject x session)x node x node)
    in the network.
    Returns a dataframe containing the results of kfolds cross-validation,
    including the indices of train and test samples, the resulting p-value and largest connected component,
    the accuracy of the network in predicting group belonging in the test samples (using logistic regression),
    the parameter estimates from each regression, and the model object from each regression. 
    from a BIDS derivative folder. Optionally returns a subject x session dataframe
    of confound measures (e.g., motion averages) and/or a node^2 x (subject x session) 
    array of vectorized upper triangles of those correlation mat
    Parameters
    ----------
    matrices : numpy array of shape (p, n, n)
        Represents the link strengths of the graphs (i.e., functional connectivity). 
        Assumed to be an array of symmetric matrices.
    outcome : list-like of shape (p,)
        Y-value to be predicted with connectivity
    confounds : list-like of shape (p,m)
        Covariates, included as predictors in model.
    alpha : float
        Type-I error (i.e., false positive) rate, for outcome-related edge detection.
    predict : bool
        If True, bypasses `permutations` parameter and only runs edge detection + component identification.
        Used for NBS-Predict.
    permutations : int
        If `predict=False`, specifies the number of permutations run to create a null distribution
        for estimating the significance of the connected component size. Recommended 10,000.
    stratified : bool or list-like of shape (p,)
        If `predict=True` and there are groups that should be equally sampled across k-fold 
        cross-validation, input should be a list of group belonging (i.e., one label per participant).

    Returns
    -------
    S1 : Pandas dataframe
        A binary matrix denoting the largest connected component.
    pval : float
        If `predict=False`, denotes the significance of the largest connected component.
    perms : numpy array of shape (permutations,)
        If `predict=False`, largest connected component size per permutation.
    '''
    # need to do a mass-univariate test at every edge
    # and retain significant edges
    # then find the largest connected component
    # and, if not predict, build a null distribution
    n = matrices.shape[:-1]
    ndims = len(matrices.shape)
    
    # vectorize_corrmats returns p x n^2
    # we want to run pynbs per edge
    # so vectorized edges must be transposed
    
    exog = np.hstack((outcome, confounds))
    exog = sm.add_constant(exog, prepend=False)
    # turn matrices into vectorized upper triangles
    if ndims > 2:
        edges = vectorize_corrmats(matrices)
    else:
        edges = matrices.copy()
    edges = edges.T
    
    # run an ols per edge
    # create significancs matrix for predictor of interest (outcome)
    # 1 if edge is significantly predicted by outcome
    # 0 if it's not
    sig_edges = []
    for i in range(0, edges.shape[0]):
        # statsmodels for regressing predictors on edges
        mod = sm.OLS(edges[i,:], exog, hasconst=True)
        results = mod.fit()
        edge_pval = results.pvalues[0]
        
        # build binary significance edge vector
        if edge_pval < alpha:
            sig_edges.append(1)
        else:
            sig_edges.append(0)
    
    # find largest connected component of sig_edges
    # turn sig_edges into an nxn matrix first
    sig_matrix = undo_vectorize(sig_edges) # need to write this function
    matrix = nx.from_numpy_array(sig_matrix)
    
    #use networkX to find connected components
    comps = nx.connected_components(matrix)
    
    # rearrange networkx output into an array of matrices, S
    S = [matrix.subgraph(c).copy() for c in comps]
    # find size of each connected component, s in S
    size = np.asarray([s.number_of_edges() for s in S])
    (max_comp, ) = np.where(size == max(size))
    largest_comp_size = max(size)
    print(f'Connected component has {largest_comp_size} edges.')

    # retain size of largest connected component 
    # for NBS permutation-based significance testing
    max_comp = max_comp[0]

    # pull the subgraph with largest number of nodes
    # i.e., the largest connected component
    G = S[max_comp]

    # grab list of nodes in largest connected component
    nodes = list(G.nodes)
    
    unused_nodes = list(set(matrix.nodes) - set(nodes))
    S1 = nx.to_pandas_adjacency(G, nodelist=nodes)

    # add empty edges for unused nodes
    # bc NBS-Predict needs all nodes for
    # the eventual weighted average
    # and NBS might need all nodes for easier
    # plotting in brain space
    for i in unused_nodes:
        S1.loc[i] = 0
        S1[i] = 0

    S1.sort_index(axis=0, inplace=True)
    S1.sort_index(axis=1, inplace=True)
    
    # permutation testing to create a null distribution of max component size
    # only for regular NBS, -Predict doesn't need this
    if predict == False:
        perms = np.zeros((permutations,))
        hit = 0
        rng = np.random.default_rng()
        exog_copy = exog.copy()
        for i in range(0, permutations):
            # shuffle outcome order
            rng.shuffle(exog_copy, axis=0)
            #print(exog_copy)
            perm_edges = []
            for j in range(0, edges.shape[0]):
                # statsmodels for regressing predictors on edges
                mod = sm.OLS(edges[j,:], exog_copy, hasconst=False)
                results = mod.fit()
                edge_pval = results.pvalues[0]
                
                if edge_pval < alpha:
                    perm_edges.append(1)
                else:
                    perm_edges.append(0)
            #print(np.sum(perm_edges))
            # find largest connected component of sig_edges
            # turn sig_edges into an nxn matrix first
            perm_matrix = undo_vectorize(perm_edges) # need to write this function
            perm_nx = nx.from_numpy_array(perm_matrix)

            comps = nx.connected_components(perm_nx)

            S = [perm_nx.subgraph(c).copy() for c in comps]
            perm_size = np.asarray([s.number_of_edges() for s in S])
            (max_comp, ) = np.where(perm_size == max(perm_size))
            #print(perm_size, max_comp)

            # retain for null distribution
            perms[i] = max(perm_size)
            if i % 10 == 0:
                print(f'p-value is {np.size(np.where(perms >= largest_comp_size)) / permutations} as of permutation {i}')
            
            # bctpy nbs code uses hit to mark progress across permutations
            # prob not necessary?
        
        # bctpy calcs pval for all components, not just largest?
        # but I don't think that's relevant for the og implimentation of nbs?
        pval = np.size(np.where(perms >= largest_comp_size)) / permutations
        print(largest_comp_size, permutations, pval)
        
        return pval, S1, perms
    else:
        return S1

def kfold_nbs(matrices, outcome, confounds, alpha, tail='both', groups=None, n_splits=10, n_iterations=10, k=1000, shuffle=False, fig_dir=None):
    """Calculates the Network Based Statistic (Zalesky et al., 20##) on connectivity matrices provided
    of shape ((subject x session)x node x node)
    in the network.
    Returns a dataframe containing the results of kfolds cross-validation,
    including the indices of train and test samples, the resulting p-value and largest connected component,
    the accuracy of the network in predicting group belonging in the test samples (using logistic regression),
    the parameter estimates from each regression, and the model object from each regression. 
    from a BIDS derivative folder. Optionally returns a subject x session dataframe
    of confound measures (e.g., motion averages) and/or a node^2 x (subject x session) 
    array of vectorized upper triangles of those correlation mat
    Parameters
    ----------
    matrices : numpy array of shape (p, n, n)
        Represents the link strengths of the graphs. Assumed to be
        an array of symmetric matrices.
    outcome : list-like of shape (p,)
        Y-value to be predicted with connectivity
    
    Returns
    -------
    cv_results : Pandas dataframe
        Includes the results of each cross-validation loop
        the input matrices.
    """
    edges = vectorize_corrmats(matrices)
    #print(edges.shape)
    index = list(range(0,n_splits * n_iterations))

    cv_results = pd.DataFrame(index=index, 
                            columns=['split',  
                                    'pval', 
                                    'score',
                                    'component',
                                    'coefficient_matrix',
                                    'coefficient_vector',
                                    'model'])
    if groups is not None:
        cv = RepeatedStratifiedKFold(n_splits=n_splits,
                                    n_repeats=n_iterations)
        df = groups.shape[0] - 2
    else:
        cv = RepeatedKFold(n_splits=n_splits, 
                        n_repeats=n_iterations)
        df = edges.shape[0] - 1
    
    if tail == 'both':
        alpha = 0.01
    else:
        alpha = 0.005
    t_threshold = t.ppf(1 - alpha, df=df)
    
    if matrices.shape[0] != matrices.shape[1]:
        if matrices.shape[1] == matrices.shape[2]:
            num_node = matrices.shape[1]
            matrices = np.moveaxis(matrices, 0, -1)
        else:
            raise ValueError(f'Matrices of shape {matrices.shape}',
                             'requires matrices of shape (subject x session) x node x node',
                             'or node x node x (subject x session).')
    else:
        num_node = matrices.shape[0]
    upper_tri = np.triu_indices(num_node, k=1)
    
    i = 0
    manager = enlighten.get_manager()
    ticks = manager.counter(total=n_splits * n_iterations, desc='Progress', unit='folds')
    for train_idx, test_idx in cv.split(edges, outcome, groups=groups):
        cv_results.at[i, 'split'] = (train_idx, test_idx)
        # all of this presumes the old bctpy version of nbs
        # irrelevant for pynbs
        #train_a_idx = [m for m in train_idx if outcome[m] == 0]
        #train_b_idx = [m for m in train_idx if outcome[m] == 1]
        #assert len(train_a_idx) == len(train_b_idx)
        #train_a = matrices[:,:,train_a_idx]
        #train_b = matrices[:,:,train_b_idx]
        #print(train_a.shape, train_b.shape)
        
        # separate edges & covariates into 
        train_y = outcome[train_idx]
        test_y = outcome[test_idx]

        pval, adj, _ = pynbs(matrices, outcome, confounds, alpha, predict=False, permutations=10000)
        pval, adj, _ = bct.nbs_bct(train_a,
                                train_b,
                                t_threshold,
                                k=k,
                                tail=tail)
        cv_results.at[i, 'pval'] = pval
        cv_results.at[i, 'component'] = adj

        nbs_vector = adj[upper_tri]
        mask = nbs_vector == 1
        train_features = edges[train_idx, :].T[mask]
        test_features = edges[test_idx, :].T[mask]

        regressor = LogisticRegression(max_iter=1000)
        model = regressor.fit(X=train_features.T, y=train_y)
        cv_results.at[i, 'model'] = model
        score = model.score(X=test_features.T, y=test_y)
        cv_results.at[i, 'score'] = score

        m = 0
        param_vector = np.zeros_like(nbs_vector)
        for l in range(0, nbs_vector.shape[0]):
            if nbs_vector[l] == 1.:
                param_vector[l] = model.coef_[0,m]
                m+=1
            else:
                pass
        X = np.zeros_like(adj)
        X[np.triu_indices(X.shape[0], k=1)] = param_vector
        X = X + X.T
        cv_results.at[i, 'coefficient_matrix'] = X
        cv_results.at[i, 'coefficient_vector'] = param_vector
        i += 1
        ticks.update()
    return cv_results