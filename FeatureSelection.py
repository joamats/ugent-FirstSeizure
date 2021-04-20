import pandas as pd
import numpy as np
from Pickle import getPickleFile
from DataAssessment import fts_correlation_matrix

#%%

dataset = getPickleFile('../3_ML_Data/128Hz/dataset')
fts_names = getPickleFile('../3_ML_Data/128Hz/featuresNames')
labels_names = getPickleFile('../3_ML_Data/128Hz/labelsNames')

#%%

def eliminate_corr_fts(dataset, fts_names, th=0.9):

    X_tr = dataset['X_tr']
    data = pd.DataFrame(data=X_tr, columns=fts_names)
    corr = fts_correlation_matrix(dataset, fts_names, k_features=None)
    
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if corr.iloc[i,j] >= th:
                if columns[j]:
                    columns[j] = False
                    
    fts_names_new = data.columns[columns]
    X_tr_new = data[fts_names_new].to_numpy()
    
    dataset['X_tr'] = X_tr_new
    
    # to be done with the test set
    # # do the same for the validation set
    # X_val = dataset['X_val'] 
    # data = pd.DataFrame(data=X_val, columns=fts_names)
    # X_val_new = data[fts_names_new].to_numpy()
    # dataset['X_val'] = X_val_new
    
    
    return dataset, fts_names_new
