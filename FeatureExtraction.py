import mne
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from PreProcessing import get_ica_template, eeg_preprocessing, clean_epochs
from Pickle import createPickleFile, getPickleFile

# #%% Run
# filenames = pd.read_excel('Metadata_train.xlsx')['Filename']
# icas = get_ica_template(filenames[0])

# imcohs = []
# plvs = []
# reject_logs = []

# for filename in filenames:
#     epochs = eeg_preprocessing(filename, icas, plot = False)
#     epochs, reject_log = clean_epochs(epochs)
#     reject_logs.append(reject_log)
    
#     if not all(reject_log.bad_epochs):
    
#         imcoh = mne.connectivity.spectral_connectivity(epochs, method = "imcoh", 
#                                  sfreq = 256, faverage=True, verbose = False)
           
#         plv = mne.connectivity.spectral_connectivity(epochs, method = "plv", 
#                                  sfreq = 256, faverage=True, verbose = False)
           
#         imcohs.append(imcoh)
#         plvs.append(plv)

# #%% Save Measures
# epochs=getPickleFile('../PreProcessed_Data/110914B-D')

#%% Mutual information
filenames = pd.read_excel('Metadata_train.xlsx')['Filename']



for filename in filenames[0:1]:
    epochs=getPickleFile('../PreProcessed_Data/' + filename)
    mutual_infos= np.zeros((np.shape(epochs._data)[1], np.shape(epochs._data)[1], 1))
                
    for channel_1 in range (np.shape(epochs._data)[1]):
        for channel_2 in range (channel_1, np.shape(epochs._data)[1]):
            all_mi=[]
            for singleEpoch in range (np.shape(epochs._data)[0]):
                x=epochs._data[singleEpoch][channel_1]
                x=x.reshape(-1,1)
                y=epochs._data[singleEpoch][channel_2]
                # y=y.reshape(-1,1)
                mi=mutual_info_regression(x, y, random_state=42)
                all_mi.append(mi)
            mutual_infos[channel_1][channel_2][0]=np.mean(all_mi)
                
                
                
                
                
                