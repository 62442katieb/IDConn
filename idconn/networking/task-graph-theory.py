import numpy as np
import pandas as pd
from os.path import join
from nilearn.connectome import ConnectivityMeasure
import bct
import datetime

correlation_measure = ConnectivityMeasure(kind="correlation")

index = pd.MultiIndex.from_product(
    [subjects, sessions, tasks, conds, masks],
    names=["subject", "session", "task", "condition", "mask"],
)

df = pd.DataFrame(
    columns=["efficiency", "charpath", "modularity", "assortativity", "transitivity"],
    index=index,
    dtype=np.float64,
)

for subject in subjects:
    for session in sessions:
        for task in tasks.keys():
            for i in np.arange(0, len(tasks[task][0]["conditions"])):
                conditions = tasks[task][0]["conditions"]
                for mask in masks:
                    try:
                        lab_notebook.at[
                            (subject, session, task, conds[i], mask), "start"
                        ] = str(datetime.datetime.now())
                        corrmat = np.genfromtxt(
                            join(
                                sink_dir,
                                sesh[session],
                                subject,
                                "{0}-session-{1}_{2}-{3}_{4}-corrmat.csv".format(
                                    subject, session, task, conditions[i], mask
                                ),
                            ),
                            delimiter=" ",
                        )

                        ge_s = []
                        cp_s = []
                        md_s = []
                        at_s = []
                        tr_s = []
                        for p in np.arange(kappa_upper, kappa_lower, 0.01):
                            ntwk = []
                            thresh = bct.threshold_proportional(corrmat, p, copy=True)

                            # network measures of interest here
                            # global efficiency
                            ge = bct.efficiency_wei(thresh)
                            ge_s.append(ge)

                            # characteristic path length
                            cp = bct.charpath(thresh)
                            cp_s.append(cp[0])

                            # modularity
                            md = bct.modularity_louvain_und(thresh)
                            md_s.append(md[1])

                            # network measures of interest here
                            # global efficiency
                            at = bct.assortativity_wei(thresh)
                            at_s.append(at)

                            # modularity
                            tr = bct.transitivity_wu(thresh)
                            tr_s.append(tr)

                        df.at[
                            (subject, session, task, conds[i], mask), "assortativity"
                        ] = np.trapz(ge_s, dx=0.01)
                        df.at[
                            (subject, session, task, conds[i], mask), "transitivity"
                        ] = np.trapz(md_s, dx=0.01)
                        df.at[
                            (subject, session, task, conds[i], mask), "efficiency"
                        ] = np.trapz(ge_s, dx=0.01)
                        df.at[
                            (subject, session, task, conds[i], mask), "charpath"
                        ] = np.trapz(cp_s, dx=0.01)
                        df.at[
                            (subject, session, task, conds[i], mask), "modularity"
                        ] = np.trapz(md_s, dx=0.01)

                        # df.to_csv(join(sink_dir, 'resting-state_graphtheory_shen+craddock.csv'), sep=',')
                    except Exception as e:
                        print(e, subject, session)
                        
