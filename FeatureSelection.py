import pandas as pd
import numpy as np
from Pickle import getPickleFile
from DataAssessment import fts_correlation_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#%%

def eliminate_corr_fts(dataset, fts_names, th=0.95):

    X_tr = dataset['X_tr']
    data = pd.DataFrame(data=X_tr, columns=fts_names)
    
    # feature normalization for correlation matrix
    norm_scaler = StandardScaler(with_mean=True, with_std=True)
    minMax_scaler = MinMaxScaler(feature_range=(0,1))
    
    X_tr_norm = norm_scaler.fit_transform(X_tr)
    X_tr_norm = minMax_scaler.fit_transform(X_tr_norm)
    
    corr = pd.DataFrame(data=X_tr_norm, columns=fts_names).corr() 
    
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if corr.iloc[i,j] >= th:
                if columns[j]:
                    columns[j] = False
                    
    fts_names_new = data.columns[columns]
    dataset['X_tr'] = data[fts_names_new].to_numpy()
        
    # do the same for the test set
    X_ts = dataset['X_ts'] 
    data = pd.DataFrame(data=X_ts, columns=fts_names)
    dataset['X_ts'] = data[fts_names_new].to_numpy()
    
    
    return dataset, fts_names_new
