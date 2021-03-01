# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 14:58:06 2021

@author: Guilherme
"""
import mne
import pandas as pd
from PreProcessing import get_ica_template, eeg_preprocessing


#%% Run
filenames = pd.read_excel('Metadata_train.xlsx')['Filename']
icas = get_ica_template(filenames[0])

imcohs = []

for filename in filenames:
   epoch = eeg_preprocessing(filename, icas, plot = False)
   imcoh = mne.connectivity.spectral_connectivity(epoch, method = "imcoh", 
                                sfreq = 256, faverage=True, verbose = False )
   imcohs.append(imcoh)