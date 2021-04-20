import seaborn as sb
import pandas as pd
import numpy as np
from BestRankedFeatures import best_ranked_features
from Pickle import getPickleFile
from matplotlib import pyplot as plt

def fts_correlation_matrix(dataset, fts_names, k_features=None):

    X_tr = dataset['X_tr']

    if k_features is None:
        X_df = pd.DataFrame(data=X_tr, columns=fts_names)
        corr_df = X_df.corr()
                
    else:
        best_fts = best_ranked_features(dataset, fts_names, k_features)
        best_idxs = best_fts.index
        # filtered df with best features only
        X_df = pd.DataFrame(data=X_tr[:,best_idxs], columns=best_fts['fts_names'])
        
        corr_df = X_df.corr()
        plt.figure()
        sb.heatmap(corr_df, annot=True, cmap="Blues")
    
    return corr_df
    
    

