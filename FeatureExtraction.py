import mne
import pandas as pd
from PreProcessing import get_ica_template, eeg_preprocessing, clean_epochs
from Pickle import createPickleFile, getPickleFile

#%% Run
filenames = pd.read_excel('Metadata_train.xlsx')['Filename']

imcohs = []
plvs = []
pdcs = []

for filename in filenames[0:1]:
    epochs = getPickleFile(filename)
    
    # guarantee epochs are not empty
    # if epochs._data.shape[0] != 0:
    
    # imcoh = mne.connectivity.spectral_connectivity(epochs, method = "imcoh", 
    #                          sfreq = 256, faverage=True, verbose = False)
    # imcohs.append(imcoh)
       
    # plv = mne.connectivity.spectral_connectivity(epochs, method = "plv", 
    #                          sfreq = 256, faverage=True, verbose = False)    
    # plvs.append(plv)
    
         
    
    

#%% Save Measures

createPickleFile(imcohs, '../Features/' + 'IMCOH')
createPickleFile(plvs, '../Features/' + 'PLV')

