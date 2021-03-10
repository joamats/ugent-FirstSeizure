import mne
import pandas as pd
import numpy as np
import scipy
import scot
import yasa
from sklearn.feature_selection import mutual_info_regression
from PreProcessing import get_ica_template, eeg_preprocessing, clean_epochs, epochs_selection_bandpower
from Pickle import createPickleFile, getPickleFile
import scot

#%% Mutual Information

def mutual_information(epochs):
    
    mutual_infos = np.zeros((np.shape(epochs._data)[1], np.shape(epochs._data)[1], 1))
    std = np.zeros((np.shape(epochs._data)[1], np.shape(epochs._data)[1], 1))
                
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
            mutual_infos[channel_1][channel_2][0]=np.mean(all_mi)
            std[channel_1][channel_2][0]=np.std(all_mi)
    
    #transforms the 3D matrix in 2D
    mi_2D=np.zeros((19,19))
    for i in range(0,np.shape(mutual_infos)[0]):
        mi_2D[i,:]=np.matrix.transpose(mutual_infos[i,:,:])
    std_2D=np.zeros((19,19))
    for i in range(0,np.shape(std)[0]):
        std_2D[i,:]=np.matrix.transpose(std[i,:,:])
    return mi_2D, std_2D

#%% Run
filenames = pd.read_excel('Metadata_train.xlsx')['Filename']

imcohs_list = []
plvs_list = []
mis_list = []
pdcs = []
bands = {'Delta': [1, 4], 'Theta': [4, 8], 'Alpha': [8,12],
             'Beta': [12, 30], 'Global': [1,30]}

for filename in filenames[[0]]:
    saved_epochs = getPickleFile('../PreProcessed_Data/' + filename)
    bd_names, s_epochs = epochs_selection_bandpower(saved_epochs)
    imcohs = {}
    plvs = {}
    
    for k in range(5):
        
        f_min = bands[bd_names[k]][0]
        f_max = bands[bd_names[k]][1]
        
        # IMOCH
        imcoh_mean_std = []
        imcoh = mne.connectivity.spectral_connectivity(s_epochs[k], method = "imcoh", 
                                  sfreq = 256, fmin=f_min, fmax=f_max, 
                                  faverage=False, verbose=False)
        
        std = np.std(imcoh[0], axis=2)
        imcoh = list(imcoh)
        imcoh_mean_std.append(np.mean(imcoh[0], axis=2))
        imcoh_mean_std.append(std)
        imcohs[bd_names[k]] = imcoh_mean_std
        
        
        # PLV
        plv_mean_std = []
        plv = mne.connectivity.spectral_connectivity(s_epochs[k], method = "plv", 
                                  sfreq = 256, fmin=f_min, fmax=f_max,
                                  faverage=False, verbose = False)    
        std = np.std(plv[0], axis=2)
        plv = list(plv)
        plv_mean_std.append(np.mean(imcoh[0], axis=2))
        plv_mean_std.append(std)
        plvs[bd_names[k]] = plv_mean_std
        
        # MI
        if(bd_names[k] == 'Global'):
            mi, std = mutual_information(s_epochs[k])
            mis_list.append([mi,std])
               
    # Stores 1 Dict per person on a list
    imcohs_list.append(imcohs)
    plvs_list.append(plvs)
#%% Save Measures

# createPickleFile(imcohs, '../Features/' + 'IMCOH')
# createPickleFile(plvs, '../Features/' + 'PLV')
# createPickleFile(mis, '../Features/' + 'MI')
# createPickleFile(pdcs, '../Features/' + 'PDC')
              
                
                