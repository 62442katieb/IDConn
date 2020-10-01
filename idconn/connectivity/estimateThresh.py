import numpy as np
import networkx as nx
import pandas as pd
import bct 


task_networks(dset_dir, subject, session, task, event_related, conditions, runs, connectivity_metric, space, atlas, confounds)
for subject in subjects:
    print(subject)
    try:
        for i in np.arange(0,len(sessions)):
            print i
            run_cond = {}
            for task in tasks.keys():
                print task
                timing = {}
                conditions = tasks[task][0]['conditions']
                for mask in masks.keys():
                    print mask
                    for j in np.arange(0,len(conditions)):
                        #reset tau starting point
                        #calculate proportion of connections that can be retained
                        #before degree dist. ceases to be scale-free
                        tau = 0.01
                        skewness = 1
                        while abs(skewness) > 0.3:
                            w = bct.threshold_proportional(corrmat, tau)
                            skewness = skew(bct.degrees_und(w))
                            tau += 0.01
                        df.at[(subject, sessions[i], task, conds[j], mask),'k_scale-free'] = tau

                        #reset tau starting point
                        #calculate proportion of connections that need to be retained
                        #for node connectedness
                        tau = 1
                        connected = False
                        while connected == False:
                            w = bct.threshold_proportional(corrmat, tau)
                            w_nx = nx.convert_matrix.from_numpy_array(w)
                            connected = nx.algorithms.components.is_connected(w_nx)
                            tau -= 0.01
                        df.at[(subject, sessions[i], task, conds[j], mask),'k_connected'] = tau