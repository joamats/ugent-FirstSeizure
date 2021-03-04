# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 09:13:56 2021

@author: Guilherme
"""
import pandas as pd
from os import listdir
from os.path import isfile, join
import mne
import numpy as np
import pandas as pd
from autoreject import AutoReject
from Pickle import createPickleFile, getPickleFile
from PreProcessing import eeg_preprocessing, clean_epochs, get_ica_template

mypath = '../PreProcessed_Data'
filenames = pd.read_excel('Metadata_train.xlsx')['Filename']

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

filenames_folder = []
for f in onlyfiles:
    name = filenames_folder.append(f.split('.')[0])
    
bools = filenames.isin(filenames_folder)
un = filenames.duplicated()

idx = [i for i, x in enumerate(bools) if not x]

#%%
icas = get_ica_template(filenames[0])

for filename in filenames[idx]:
    epochs = eeg_preprocessing(filename, icas, plot = False)
    epochs, reject_log = clean_epochs(epochs)
    createPickleFile(epochs, '../PreProcessed_Data/' + filename)
    
