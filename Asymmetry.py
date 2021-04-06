import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#%%
def _reorder_matrix(data_array, ch_names_new):
    
    ch_names_original = ['Fz', 'Cz', 'Pz', 'Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T7', 
                          'P7', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T8', 'P8']

    ch_indexes = []
    
    # New channel order's indexes on the ch_names_original variable
    for ch in ch_names_new:
        ch_indexes.append(ch_names_original.index(ch))
    
    # Only for triangular matrices
    if np.allclose(data_array, np.tril(data_array)): 
        # makes the triangular matrix into a symmetric matrix
        data_array = data_array + data_array.T - np.diag(np.diag(data_array))
        b = np.zeros((19,19))
        
        # Change the order of the channels
        b = data_array[:,ch_indexes]
        b = b[ch_indexes,:]
        data_array = np.tril(b)
        
    else:
        b = np.zeros((19,19))
        # Change the order of the channels
        b = data_array[:,ch_indexes]
        b = b[ch_indexes,:]
    
    return pd.DataFrame(data=data_array, columns=ch_names_new)

#%%

ch_names_new = ['F8', 'T8', 'P8', 'Fz', 'Cz', 'Pz', 'Fp1', 'F3', 'C3', 'P3', 'O1',
                'F7', 'T7', 'P7', 'Fp2', 'F4', 'C4', 'P4', 'O2']

data_array = np.ones((19,19))
data_array[18,:] = 0
new_data = _reorder_matrix(data_array, ch_names_new)