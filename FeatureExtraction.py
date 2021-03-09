import mne
import pandas as pd
import numpy as np
import scipy
import scot
import yasa
from sklearn.feature_selection import mutual_info_regression
from PreProcessing import get_ica_template, eeg_preprocessing, clean_epochs
from Pickle import createPickleFile, getPickleFile
# import scot

#%% Mutual Information

def mutual_information(epochs):
    
    mutual_infos = np.zeros((np.shape(epochs._data)[1], np.shape(epochs._data)[1], 1))
                
    for channel_1 in range (1, np.shape(epochs._data)[1]):
        for channel_2 in range (channel_1):
            all_mi = []
            for singleEpoch in range (np.shape(epochs._data)[0]):
                x = epochs._data[singleEpoch][channel_1]
                x = x.reshape(-1,1)
                y = epochs._data[singleEpoch][channel_2]
                mi = mutual_info_regression(x, y, random_state=42)
                all_mi.append(mi)
            mutual_infos[channel_1][channel_2][0]=np.mean(all_mi)
    
    return mutual_infos

#%% Run
filenames = pd.read_excel('Metadata_train.xlsx')['Filename']

imcohs = []
plvs = []
mis = []
pdcs = []

for filename in filenames[0:1]:
    epochs = getPickleFile('../PreProcessed_Data/' + filename)
    # IMOCH
    imcoh = mne.connectivity.spectral_connectivity(epochs, method = "imcoh", 
                              sfreq = 256, faverage=False, verbose = False)
    imcohs.append(imcoh)
       
    # PLV
    plv = mne.connectivity.spectral_connectivity(epochs, method = "plv", 
                              sfreq = 256, faverage=False, verbose = False)    
    plvs.append(plv)
    
    # MI
    mi = mutual_information(epochs)
    mis.append(mi)
    
    # PDC
    ws = scot.Workspace({'model_order': 40}, reducedim='no_pca', fs=256, nfft=512)
    ws.set_data(epochs._data)
    ws.do_mvarica()
    pdc = ws.get_connectivity(measure_name='PDC', plot=None)
    pdcs.append(pdc)
        
#%% Save Measures

createPickleFile(imcohs, '../Features/' + 'IMCOH')
createPickleFile(plvs, '../Features/' + 'PLV')
createPickleFile(mis, '../Features/' + 'MI')
createPickleFile(pdcs, '../Features/' + 'PDC')

                
                
                
                
                