import numpy as np
import pingouin as pg
import networkx as nx
import pandas as pd
from idconn.io import vectorize_corrmats, undo_vectorize
from scipy.stats import t, pearsonr, pointbiserialr, spearmanr
import enlighten
#import bct

from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold, GridSearchCV, StratifiedKFold, KFold

from sklearn.feature_selection import f_regression, f_classif
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error

def calc_number_of_nodes(matrices):
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
    return num_node

def residualize(X, y=None, confounds=None):
    # residualize the outcome
    if confounds is not None:
        if y is not None:
            temp_y = np.reshape(y, (y.shape[0],))
            y = pg.linear_regression(confounds, temp_y)
            resid_y = y.residuals_

            # residualize features
            resid_X = np.zeros_like(X)
            #print(X.shape, resid_X.shape)
            for i in range(0, X.shape[1]):
                X_temp = X[:,i]
                #print(X_temp.shape)
                X_ = pg.linear_regression(confounds, X_temp)
                #print(X_.residuals_.shape)
                resid_X[:,i] = X_.residuals_.flatten()
            return resid_y, resid_X
        else:
            # residualize features
            resid_X = np.zeros_like(X)
            #print(X.shape, resid_X.shape)
            for i in range(0, X.shape[1]):
                X_temp = X[:,i]
                #print(X_temp.shape)
                X_ = pg.linear_regression(confounds, X_temp)
                #print(X_.residuals_.shape)
                resid_X[:,i] = X_.residuals_.flatten()
            return resid_X
    else:
        print('Confound matrix wasn\'t provided, so no confounding was done')
        
    

def pynbs(matrices, outcome, alpha=0.05, predict=False, permutations=10000):
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
        Type-I error (i.e., false positive) rate, for outcome-related edge detection. Default = 0.05
    predict : bool
        If True, bypasses `permutations` parameter and only runs edge detection + component identification.
        Used for NBS-Predict.
    permutations : int
        If `predict=False`, specifies the number of permutations run to create a null distribution
        for estimating the significance of the connected component size. Recommended 10,000.
    
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
    #n = matrices.shape[:-1]
    ndims = len(matrices.shape)
    
    # vectorize_corrmats returns p x n^2

    # turn matrices into vectorized upper triangles
    if ndims > 2:
        edges = vectorize_corrmats(matrices)
    else:
        edges = matrices.copy()
    #print(edges.shape)
    
    
    #edges = edges.T
    
    # run an ols per edge
    # create significancs matrix for predictor of interest (outcome)
    # 1 if edge is significantly predicted by outcome
    # 0 if it's not
    
    if len(np.unique(outcome)) < 5:
        (f, p) = f_classif(X=edges, y=outcome)
    else:
        (f, p) = f_regression(X=edges, y=outcome, center=False)
    sig_edges = np.where(p < alpha, 1, 0)
    
    # find largest connected component of sig_edges
    # turn sig_edges into an nxn matrix first
    sig_matrix = undo_vectorize(sig_edges) # need to write this function
    matrix = nx.from_numpy_array(sig_matrix)
    
    #use networkX to find connected components
    largest_cc = max(nx.connected_components(matrix), key=len)
    G0 = matrix.subgraph(largest_cc)
    #print(G0)
    
    # retain size of largest connected component 
    # for NBS permutation-based significance testing
    max_comp = G0.number_of_edges()
    #print(f'Connected component has {max_comp} edges.')    

    # pull the subgraph with largest number of nodes
    # i.e., the largest connected component
    
    # grab list of nodes in largest connected component
    nodes = list(G0.nodes)
    
    unused_nodes = list(set(matrix.nodes) - set(nodes))
    S1 = nx.to_pandas_adjacency(G0, nodelist=nodes)

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
        rng = np.random.default_rng()
        outcome_copy = outcome.copy()
        for i in range(0, permutations):
            # shuffle outcome order
            rng.shuffle(outcome_copy, axis=0)
            #print(outcome_copy)
            
            if len(np.unique(outcome)) < 5:
                (f1, p1) = f_classif(edges, outcome_copy)
            else:
                (f1, p1) = f_regression(edges, outcome_copy, center=False)
            
            perm_edges = np.where(p1 < alpha, 1, 0)
            
            #print(np.sum(perm_edges))
            # find largest connected component of sig_edges
            # turn sig_edges into an nxn matrix first
            perm_matrix = undo_vectorize(perm_edges) # need to write this function
            perm_nx = nx.from_numpy_array(perm_matrix)

            largest_cc = max(nx.connected_components(perm_nx), key=len)
            S = perm_nx.subgraph(largest_cc)

            perm_comp_size = S.number_of_edges()
            

            # retain for null distribution
            perms[i] = perm_comp_size
            if i == 0:
                pass
            elif i % 100 == 0:
                print(f'p-value is {np.round(np.sum(np.where(perms >= max_comp, 1, 0)) / i, 3)} as of permutation {i}')
            
            # bctpy nbs code uses hit to mark progress across permutations
            # prob not necessary?
        
        # bctpy calcs pval for all components, not just largest?
        # but I don't think that's relevant for the og implimentation of nbs?
        pval = np.size(np.where(perms >= max_comp)) / permutations
        print(max_comp, permutations, pval)
        
        return pval, S1, perms
    else:
        return S1

def kfold_nbs(matrices, outcome, confounds=None, alpha=0.05, groups=None, n_splits=10, n_iterations=10):
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
    confounds : list-like
        Columns in `participants.tsv` to be regressed out of connectivity and outcome 
        data in each CV fold (per recommendation from Snoek et al., 2019).
    alpha : float
        Proportion of type II errors (i.e., false positives) we're willing to put up with. 
        This is the upper limit for pvalues in the edge detection process.
    groups : list-like of shape (p,)
        Grouping variable - currently only works for 2 groups. Will enforce stratified k-fold CV.
        Currently intended for use where grouping variable is the outcome of interest, assumed by StratifiedKFold.
        NEED TO FIX THIS: ALLOW THE CASE WHERE GROUPING VAR != OUTCOME VAR
    n_splits : int
        Value of K for K-fold cross-validation. Will split data into K chunks, train on K-1 chunks and test on the Kth.
    n_iterations : int
        Number of times to run K-fold cross-validation. More times = more stable results.
    
    Returns
    -------
    weighted_average : Pandas dataframe
        Includes the average of all largest components across folds and iterations, weighted by
        their prediction performance (i.e., accuracy for binary outcome, correlation for continuous).
        Could be used for out-of-sample prediction, once thresholded and binarized.
    cv_results : Pandas dataframe
        Includes the results of each cross-validation loop 
        (e.g., predictive performance, data split, largest connected component per fold per iteration).
    """
    edges = vectorize_corrmats(matrices)
    #print(edges.shape)
    #print(edges.shape)
    index = list(range(0,n_splits * n_iterations))

    cv_results = pd.DataFrame(index=index, 
                            columns=['split',  
                                    #'pval', 
                                    'score',
                                    'component',
                                    'coefficient_matrix',
                                    'coefficient_vector',
                                    'model'])
    if groups is not None:
        cv = RepeatedStratifiedKFold(n_splits=n_splits,
                                    n_repeats=n_iterations)
        split_y = groups
        
    else:
        cv = RepeatedKFold(n_splits=n_splits, 
                        n_repeats=n_iterations) 
        split_y = outcome   
    
    num_node = calc_number_of_nodes(matrices)
    #print(num_node)
    #if matrices.shape[0] != matrices.shape[1]:
    #    if matrices.shape[1] == matrices.shape[2]:
    #        num_node = matrices.shape[1]
            #matrices = np.moveaxis(matrices, 0, -1)
    #    else:
    #        raise ValueError(f'Matrices of shape {matrices.shape}',
                             #'requires matrices of shape (subject x session) x node x node',
                             #'or node x node x (subject x session).')
    #else:
    #    num_node = matrices.shape[0]
    upper_tri = np.triu_indices(num_node, k=1)
    
    i = 0
    manager = enlighten.get_manager()
    ticks = manager.counter(total=n_splits * n_iterations, desc='Progress', unit='folds')
    for train_idx, test_idx in cv.split(edges, split_y):
        scaler = StandardScaler()
        cv_results.at[i, 'split'] = (train_idx, test_idx)
        
        #assert len(train_a_idx) == len(train_b_idx)
        if np.unique(outcome).shape[0] == 2:
            regressor = LogisticRegression(l1_ratio=0.25, max_iter=1000, penalty='elasticnet', solver='saga')
        else:
            regressor = ElasticNet(l1_ratio=0.25, max_iter=1000)

        train_y = outcome[train_idx]
        test_y = outcome[test_idx]

        train_edges = edges[train_idx, :]
        test_edges = edges[test_idx, :]
        
        if confounds is not None:
            train_confounds = confounds.values[train_idx]
            test_confounds = confounds.values[test_idx]
            #print(train_edges.shape, train_confounds.shape, train_y.shape)
            
            # residualize the edges and outcome
            if np.unique(outcome).shape[0] == 2:
                train_edges = residualize(train_edges,train_confounds)
                test_edges = residualize(test_edges, test_confounds)
            elif np.unique(outcome).shape[0] > 3:
                train_y, train_edges = residualize(train_edges, train_y, train_confounds)
                test_y, test_edges = residualize(test_edges, test_y, test_confounds)
        else:
            pass
        
        train_edges = scaler.fit_transform(train_edges)
        test_edges = scaler.fit_transform(test_edges)

        if np.unique(outcome).shape[0] == 2:
            pass
        else:
            train_y = scaler.fit_transform(train_y.reshape(-1, 1))
            test_y = scaler.fit_transform(test_y.reshape(-1, 1))
        
        # perform NBS wooooooooo
        # note: output is a dataframe :)
        # PYNBS SHOULD NOT DO CONFOUND REGRESSION?
        adj = pynbs(train_edges, train_y, alpha, predict=True)
        #print(adj.shape, adj.ndim, adj[0].shape, upper_tri)
        
        #cv_results.at[i, 'pval'] = pval
        cv_results.at[i, 'component'] = adj.values
        
        # in the event of no edges significantly related to <outcome>
        #print(sum(sum(adj.values)), '\n', adj.values.shape)
        if sum(sum(adj.values)) > 0:
            # grab the values of the adjacency matrix that are just in the upper triangle
            # so you don't have repeated edges
            # returns (n_edges, )
            nbs_vector = adj.values[upper_tri]
            #print(nbs_vector.shape)
            # use those to make a "significant edges" mask
            mask = nbs_vector == 1.0

            # grab only the significant edges from testing and training sets of edges
            # for use as features in the predictive models
            # these are already residualized
            #print(train_edges.shape)
            # returns (n_edges, samples)
            train_features = train_edges.T[mask]
            test_features = test_edges.T[mask]

            train_features = scaler.fit_transform(train_features.T)
            test_features = scaler.fit_transform(test_features.T)
            #print(np.ravel(train_y))
            # train model predicting outcome from brain (note: no mas covariates)
            model = regressor.fit(X=train_features, y=np.ravel(train_y))
            cv_results.at[i, 'model'] = model
            
            # score that model on the testing data
            # if logistic regression: score = mean accuracy
            # if linear regression: score = coefficient of determination (R^2)
            # both from 0 (low) to 1 (high)
            score = model.score(X=test_features, y=np.ravel(test_y))
            cv_results.at[i, 'score'] = score
            #print(model.coef_.shape)

            m = 0
            param_vector = np.zeros_like(nbs_vector)
            for l in range(0, nbs_vector.shape[0]):
                if nbs_vector[l] == 1.:
                    ###
                    # NEEDS IF STATEMENT BC LOGISTIC AND LINEAR HAVE DIFFERENT COEF_ SHAPES
                    if np.unique(outcome).shape[0] == 2:
                        param_vector[l] = model.coef_[0,m]
                    else:
                        param_vector[l] = model.coef_[m]   
                    m+=1
                else:
                    pass
            X = undo_vectorize(param_vector, num_node=num_node)
            cv_results.at[i, 'coefficient_matrix'] = X
            cv_results.at[i, 'coefficient_vector'] = param_vector
            i += 1
        else:
            pass
        ticks.update()
    # calculate weighted average
    #print(cv_results['score'])
    weighted_stack = cv_results.at[0, 'component'] * cv_results.at[0, 'score']
    #print(weighted_stack.shape)
    for j in index[1:]:
        #print(cv_results.at[j, 'score'])
        if cv_results.at[j, 'score'] > 0:
            weighted = cv_results.at[j, 'component'] * cv_results.at[j, 'score']
            weighted_stack = np.dstack([weighted_stack, weighted])
        else:
            pass
        #print(weighted_stack.shape, weighted.shape)
    weighted_average = np.mean(weighted_stack, axis=-1)
    return weighted_average, cv_results