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
epochss = []
rejects = []

for filename in filenames[0:9]:
   epochs = eeg_preprocessing(filename, icas, plot = False)
   epochs, reject = clean_epochs(epochs)
   epochss.append(epochs)
   rejects.append(reject)
   imcoh = mne.connectivity.spectral_connectivity(epochs, method = "imcoh", 
                                                 sfreq = 256, faverage=True, verbose = False )
   imcohs.append(imcoh)