import pandas as pd
from PreProcessing import  get_ica_template, eeg_preprocessing, clean_epochs
from Pickle import createPickleFile, getPickleFile

filenames = pd.read_excel('Metadata_train.xlsx')['Filename']

#%%
icas = get_ica_template(filenames[0])

for filename in filenames:
    epochs = eeg_preprocessing(filename, icas, plot=False)
    epochs, _ = clean_epochs(filename, epochs, plot=False)
    createPickleFile(epochs, '../1_PreProcessed_Data/' + filename)

#%% Downsample to 128Hz

for filename in filenames:
    epochs = getPickleFile('../1_PreProcessed_Data/' + filename)
    epochs.resample(sfreq=128)
    createPickleFile(epochs, '../1_PreProcessed_Data/128Hz/' + filename)

