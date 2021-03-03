import mne
import pandas as pd
from PreProcessing import get_ica_template, eeg_preprocessing, clean_epochs
from Pickle import createPickleFile, getPickleFile
import scot

#%% Run
filenames = pd.read_excel('Metadata_train.xlsx')['Filename']

imcohs = []
plvs = []
pdcs = []
pdcs = []

for filename in filenames[0:1]:
    epochs = getPickleFile('../PreProcessed_Data/' + filename)
    
    # IMOCH
    imcoh = mne.connectivity.spectral_connectivity(epochs, method = "imcoh", 
                              sfreq = 256, faverage=True, verbose = False)
    imcohs.append(imcoh)
       
    # PLV
    plv = mne.connectivity.spectral_connectivity(epochs, method = "plv", 
                              sfreq = 256, faverage=True, verbose = False)    
    plvs.append(plv)
    
    # PDC
    ws = scot.Workspace({'model_order': 40}, reducedim='no_pca', fs=256, nfft=1024)
    ws.set_data(epochs._data)
    ws.do_mvarica()
    pdc = ws.get_connectivity(measure_name='PDC', plot=None)
    pdcs.append(pdc)
        

#%% Save Measures

createPickleFile(imcohs, '../Features/' + 'IMCOH')
createPickleFile(plvs, '../Features/' + 'PLV')
createPickleFile(pdcs, '../Features/' + 'PDC')

