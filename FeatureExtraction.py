import mne
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from PreProcessing import epochs_selection_bandpower
from Pickle import createPickleFile, getPickleFile

#%% Mutual Information

def mutual_information(epochs):
    
    mutual_infos = np.zeros((np.shape(epochs._data)[1], np.shape(epochs._data)[1], 1))
    std = np.zeros((np.shape(epochs._data)[1], np.shape(epochs._data)[1], 1))
    
    # computes for each channel combination the averaged MI
    for channel_1 in range (1, np.shape(epochs._data)[1]):
        for channel_2 in range (channel_1):
            all_mi = []
            for singleEpoch in range (np.shape(epochs._data)[0]):
                x = epochs._data[singleEpoch][channel_1]
                x = x.reshape(-1,1)
                y = epochs._data[singleEpoch][channel_2]
                mi = mutual_info_regression(x, y, random_state=42)
                all_mi.append(mi)
            #Saves the all the MI mean and std in the right position
            mutual_infos[channel_1][channel_2][0] = np.mean(all_mi)
            std[channel_1][channel_2][0] = np.std(all_mi)
    
    # transforms the 3D matrix in 2D    
    m_mi = _3D_to_triangular(mutual_infos)
    std_mi = _3D_to_triangular(std)
        
    return m_mi, std_mi

#%% Auxiliary function

def _3D_to_triangular(m_3D):
    n = np.shape(m_3D)[0]
    m_2D = np.zeros((n,n))
    
    for i in range(n):
        m_2D[i,:] = np.matrix.transpose(m_3D[i,:,:])
    
    return m_2D

#Computes mean and standard deviation for the feature data matrix
def _compute_feature_mean_std(feature_data):
    feature_data_mean_std = []
    feature_data_mean_std.append(np.mean(feature_data, axis=2))
    feature_data_mean_std.append(np.std(feature_data, axis=2))
    
    return feature_data_mean_std

#%% Run
filenames = pd.read_excel('Metadata_train.xlsx')['Filename']

imcohs_list = []
plvs_list = []
mis_list = []

bands = {'Global': [2.5,30], 'Delta': [2.5, 4], 'Theta': [4, 8],
         'Alpha': [8,12], 'Beta': [12, 30]}

for filename in filenames:
    saved_epochs = getPickleFile('../PreProcessed_Data/' + filename)
    bd_names, s_epochs = epochs_selection_bandpower(saved_epochs)
    imcohs = {}
    plvs = {}
    
    for k in range(5):
        f_min = bands[bd_names[k]][0]
        f_max = bands[bd_names[k]][1]
        
        # IMCOH
        imcoh_mean_std = []
        imcoh = mne.connectivity.spectral_connectivity(s_epochs[k], method = "imcoh", 
                                  sfreq = 256, fmin=f_min, fmax=f_max, 
                                  faverage=False, verbose = False)
        # saves on the respective bandwidth the mean and std
        imcohs[bd_names[k]] = _compute_feature_mean_std(imcoh[0])
           
        # PLV
        plv_mean_std = []
        plv = mne.connectivity.spectral_connectivity(s_epochs[k], method = "plv", 
                                  sfreq = 256, fmin=f_min, fmax=f_max,
                                  faverage=False, verbose = False)   
        # saves on the respective bandwidth the mean and std
        plvs[bd_names[k]] = _compute_feature_mean_std(plv[0])

        # MI
        if(bd_names[k] == 'Global'):
            mis_list.append(mutual_information(s_epochs[k]))
               
    # stores 1 Dict per person on a list
    imcohs_list.append(imcohs)
    plvs_list.append(plvs)
    
#%% Save Measures

createPickleFile(imcohs_list, '../Features/' + 'imcoh')
createPickleFile(plvs_list, '../Features/' + 'plv')
createPickleFile(mis_list, '../Features/' + 'mi')
              
                
                