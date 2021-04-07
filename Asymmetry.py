import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from FeatureExtraction import band_power_measures
from Pickle import getPickleFile, createPickleFile


#%% Drop channels, calculate band powers and save on pickle file
#chs_to_drop: list of channels to drop in the epochs
#hemisphere: str for the pickle file (ex: right -> bdp_right)
def bd_powers_hemisphere(chs_to_drop, hemisphere):
    filenames = pd.read_excel('Metadata_train.xlsx')['Filename']
    BDP = {}
    
    for i, filename in enumerate(filenames):
        saved_epochs = getPickleFile('../1_PreProcessed_Data/128Hz/' + filename)
        saved_epochs.drop_channels(chs_to_drop)
        
        bd_powers = band_power_measures(saved_epochs)
        BDP[filename] = bd_powers
        print(i)
                
    # save features in pickle
    createPickleFile(BDP, '../2_Features_Data/128Hz/' + 'bdp_' + hemisphere)
    
    return BDP
    
#%% Band Power Measure

filenames = pd.read_excel('Metadata_train.xlsx')['Filename']

ch_names_left = ['Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5']

ch_names_right = ['Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6']

BDP_Right = bd_powers_hemisphere(ch_names_left, 'right')
BDP_Left = bd_powers_hemisphere(ch_names_right, 'left')
        

#%%
# def _reorder_matrix(data_array, ch_names_new):
    
#     ch_names_original = ['Fz', 'Cz', 'Pz', 'Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T7', 
#                           'P7', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T8', 'P8']

#     ch_indexes = []
    
#     # New channel order's indexes on the ch_names_original variable
#     for ch in ch_names_new:
#         ch_indexes.append(ch_names_original.index(ch))
    
#     # Only for triangular matrices
#     if np.allclose(data_array, np.tril(data_array)): 
#         # makes the triangular matrix into a symmetric matrix
#         data_array = data_array + data_array.T - np.diag(np.diag(data_array))
#         b = np.zeros((19,19))
        
#         # Change the order of the channels
#         b = data_array[:,ch_indexes]
#         b = b[ch_indexes,:]
#         data_array = np.tril(b)
        
#     else:
#         b = np.zeros((19,19))
#         # Change the order of the channels
#         b = data_array[:,ch_indexes]
#         b = b[ch_indexes,:]
    
#     return pd.DataFrame(data=data_array, columns=ch_names_new)

# #%%

# ch_names_new = ['F8', 'T8', 'P8', 'Fz', 'Cz', 'Pz', 'Fp1', 'F3', 'C3', 'P3', 'O1',
#                 'F7', 'T7', 'P7', 'Fp2', 'F4', 'C4', 'P4', 'O2']

# data_array = np.ones((19,19))
# data_array[18,:] = 0
# new_data = _reorder_matrix(data_array, ch_names_new)