# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 14:58:06 2021

@author: Guilherme
"""
import mne
import pandas as pd
from PreProcessing import get_ica_template, eeg_preprocessing, clean_epochs


#%% Run
filenames = pd.read_excel('Metadata_train.xlsx')['Filename']
icas = get_ica_template(filenames[0])

imcohs = []
plvs = []
reject_logs = []

for filename in filenames:
    epochs = eeg_preprocessing(filename, icas, plot = False)
    epochs, reject_log = clean_epochs(epochs)
    reject_logs.append(reject_log)
    
    if not all(reject_log.bad_epochs):
    
        imcoh = mne.connectivity.spectral_connectivity(epochs, method = "imcoh", 
                                 sfreq = 256, faverage=True, verbose = False)
           
        plv = mne.connectivity.spectral_connectivity(epochs, method = "plv", 
                                 sfreq = 256, faverage=True, verbose = False)
           
        imcohs.append(imcoh)
        plvs.append(plv)
   