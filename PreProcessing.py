import mne
from BandpowerCorrection import bandpower_1f_correction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from autoreject import AutoReject
from Pickle import createPickleFile, getPickleFile
from scipy.fft import fft, ifft, fftfreq


#%% Imports EEG from EDF files
def import_eeg(filename):
    raw = mne.io.read_raw_edf('../0_Dataset/{}_0000.Export.edf'
                              .format(filename), preload=True, verbose=False)
    # Rename channels (if necessary)
    if 'FP1' in raw.info['ch_names']:
        raw.rename_channels({'FP1': 'Fp1', 'FP2': 'Fp2'})
    if 'T7' in raw.info['ch_names']:
        raw.rename_channels({'T7': 'T3', 'P7': 'T5', 'T8': 'T4', 'P8': 'T6'})

    # Select relevant channels: EEG only
    raw.pick_types(eeg=True, exclude=['ECG', 'EMG', 'EOG', 'SLI', 'ABD', 'ABDO', 'Status',
                                      'T1', 'T2', 'CP1', 'CP2', 'CP5', 'CP6', 'FC1', 'FC2', 'FC5', 'FC6'])
    
    # Reorder Channels if needed
    raw.reorder_channels(['Fz', 'Cz', 'Pz', 'Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5', 
                          'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6'])

    # Set average referencing
    raw.set_eeg_reference(ref_channels='average')

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
def eeg_preprocessing(filename, icas, epoch_length=2.5, plot=False):
    
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
    
    # create epochs with 2.5sec /// 5sec epoch_length
    epochs = mne.make_fixed_length_epochs(raw, duration=epoch_length, verbose = False, preload=True)
    epochs.drop_bad(flat=flat_criteria)
    
    if plot == True:
        # Plot Sensors Scalp
        orig_raw.plot_sensors(ch_type='eeg')
        # PSD Plot
        raw.plot_psd()
        # Filtered EEG Plot
        orig_raw.plot(duration=15, n_channels = 20, title="Original " + filename, remove_dc = False)
        # ICA Components in Scalp
        ica.plot_components(title="ICA Components " + filename)
        # ICA Sources Time Series
        ica.plot_sources(orig_raw, title="ICA Sources " + filename)
        # Without Artifacts EEG Plot
        raw.plot(duration=15, n_channels = 20, title="Preprocessed " + filename, remove_dc = False)
        # Epochs Plot
        epochs.plot(n_epochs=10, n_channels= 20, title="EEG " + epoch_length + "s Epochs " + filename)     
    
    return epochs

#%% Removes noisy epochs
def clean_epochs(filename, epochs, plot=False):
    ar = AutoReject(verbose=False, random_state=42, n_jobs=-1)
    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
    
    if plot == True:
        reject_log.plot_epochs(epochs, title="Clean Epochs "  + filename)
    return epochs_clean, reject_log


#%% Resample epochs from 256Hz to 128Hz

def resample_epochs(epochs, sfreq=128):
    
    return epochs.resample(sfreq)

#%% Set longitudinal bipolar montage

def set_bipolar(epochs):
    anode = ['Fp1', 'F3', 'C3', 'P3', 'Fp2', 'F4', 'C4', 'P4', 'Fp1', 'F7', 'T3', 'T5', 'Fp2', 'F8', 'T4', 'T6','Fz', 'Cz']
    cathode = ['F3', 'C3', 'P3', 'O1', 'F4', 'C4', 'P4', 'O2', 'F7', 'T3', 'T5', 'O1', 'F8', 'T4', 'T6','O2', 'Cz','Pz']
   
    epochs = mne.set_bipolar_reference(epochs, anode, cathode, drop_refs=False)
    epochs.picks = np.array(range(0,37))
    epochs.drop_channels(['Fz', 'Cz', 'Pz', 'Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6'])
    # drop signals with very low peak-to-peak
    flat_criteria = dict(eeg = 1e-6)
    epochs.drop_bad(flat=flat_criteria)

    return epochs