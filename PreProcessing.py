import mne
import yasa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from autoreject import AutoReject
from Pickle import createPickleFile, getPickleFile
from scipy.fft import fft, ifft, fftfreq


#%% Imports EEG from EDF files
def import_eeg(filename):
    raw = mne.io.read_raw_edf('../Dataset/{}_0000.Export.edf'
                              .format(filename), preload=True, verbose=False)
    # Rename channels (if necessary)
    if 'FP1' in raw.info['ch_names']:
        raw.rename_channels({'FP1': 'Fp1', 'FP2': 'Fp2'})
    if 'T7' in raw.info['ch_names']:
        raw.rename_channels({'T7': 'T3', 'P7': 'T5', 'T8': 'T4', 'P8': 'T6'})

    # Select relevant channels: EEG only
    raw.pick_types(eeg=True, exclude=['ECG','EMG', 'EOG', 'SLI', 'ABD', 'ABDO', 'Status',
                                      'T1', 'T2', 'CP1', 'CP2', 'CP5', 'CP6', 'FC1', 'FC2', 'FC5', 'FC6'])
    
    # Reorder Channels if needed
    raw.reorder_channels(['Fz', 'Cz', 'Pz', 'Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5', 
                          'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6'])
    
    # Average referencing
    raw.set_eeg_reference()
    
    # Set 10-20 montage
    ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(ten_twenty_montage)
    
    return raw

#%% Filter EEG
def filter_eeg(raw):
    # f_stop = 50
    f_low = 1
    f_high = 40
    
    # raw.notch_filter(f_stop)
    raw.filter(f_low, f_high)
    return raw

#%% Gets the file at filename, for component template purpose
def get_ica_template(filename):
    # import signal
    raw = import_eeg(filename)
    
    # filter signal
    raw = filter_eeg(raw)
    
    # fit ICA
    ica = mne.preprocessing.ICA(n_components=15, random_state=42)
    ica.fit(raw, verbose = False)
    
    icas = [ica, ica]
    
    return icas

#%% Filter, artifact removal and epochs file at filename
def eeg_preprocessing(filename, icas, plot=False):
    
    raw = import_eeg(filename)
    
    # filter signal
    raw = filter_eeg(raw)
    
    # drop signals with too low peak-to-peak
    flat_criteria = dict(eeg = 1e-6)

    # fit ICA
    ica = mne.preprocessing.ICA(n_components=15, random_state=42)
    ica.fit(raw, verbose=False)
    
    # replace new ICA in ICAS list
    icas[1] = ica
        
    # template matching
    mne.preprocessing.corrmap(icas, template=(0, 0), threshold=0.8, label='blink', plot = False, verbose = False)
    mne.preprocessing.corrmap(icas, template=(0, 1), threshold=0.8, label='blink', plot = False, verbose = False)
   
    # exclude EOG artifacts
    ica.exclude = icas[1].labels_['blink']

    # backup filtered raw eeg variable
    orig_raw = raw.copy()
    ica.apply(raw, verbose=False)
    
    # create epochs with 2sec
    epochs = mne.make_fixed_length_epochs(raw, duration=2.0, verbose = False, preload=True)
    epochs.drop_bad(flat=flat_criteria)
    
    if plot == True:
        # Plot Sensors Scalp
        orig_raw.plot_sensors(ch_type='eeg')
        # PSD Plot
        raw.plot_psd()
        # Filtered EEG Plot
        orig_raw.plot(duration=15, n_channels = 19, title="Original " + filename, remove_dc = False)
        # ICA Components in Scalp
        ica.plot_components(title="ICA Components " + filename)
        # ICA Sources Time Series
        ica.plot_sources(orig_raw, title="ICA Sources " + filename)
        # Without Artifacts EEG Plot
        raw.plot(duration=15, n_channels = 19, title="Preprocessed " + filename, remove_dc = False)
        # Epochs Plot
        epochs.plot(n_epochs=10, n_channels=19, title="EEG 2s Epochs " + filename)     
    
    return raw, epochs

#%% Removes noisy epochs
def clean_epochs(filename, epochs, plot=False):
    ar = AutoReject(verbose=False, random_state=42)
    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
    
    if plot == True:
        reject_log.plot_epochs(epochs, title="Clean Epochs "  + filename)
    
    return epochs_clean, reject_log

#%% Get bandpowers from Epochs

def epochs_selection_bandpower(epochs):
    bands = [(1, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 30, 'Beta')]
    ch = epochs.ch_names
    ms = []
    for i in range (np.shape(epochs._data)[0]):
        # compute bandpower
        bd = yasa.bandpower(data=epochs._data[i,:,:], sf=256, ch_names=ch,
                            hypno=None, relative=True, bands=bands)
        
        bd_means = bd.mean(axis=0)
        b = np.array([bd_means['Delta'], bd_means['Theta'], bd_means['Alpha'], bd_means['Beta']])
        
        var = np.var(b)
        power = bd_means['TotalAbsPow']
        
        # selection measure
        measure = np.log10(power / var * 10e9)
        ms.append(measure)
    
    return ms

# %% Run and Tests

filenames = pd.read_excel('Metadata_train.xlsx')['Filename']
icas = get_ica_template(filenames[0])

for filename in filenames[0:1]:
    _, epochs = eeg_preprocessing(filename, icas, plot=False)
    epochs, _ = clean_epochs(filename, epochs, plot=False)
    # ms = epochs_selection_bandpower(epochs)
    # plt.hist(ms)
    
#%%
# Import the module for estimating an ARMA model
from statsmodels.tsa.arima_model import ARMA

filenames = pd.read_excel('Metadata_train.xlsx')['Filename']
for filename in filenames[0:1]:
    epochs = getPickleFile('../PreProcessed_Data/' + filename)
    # Fit the data to an AR(p) for p = 0,...,100 , and save the BIC
    Bic = np.zeros(100)
    for p in range(0,100):
        mod = ARMA(epochs._data, order=(p,p))
        res = mod.fit()
        # Save BIC for AR(p)    
        Bic[p] = res.bic
        
    # Plot the BIC as a function of p
    plt.plot(range(0,100), Bic[0:100], marker='o')
    plt.xlabel('Order of AR Model')
    plt.ylabel('Bayesian Information Criterion')
    plt.show()