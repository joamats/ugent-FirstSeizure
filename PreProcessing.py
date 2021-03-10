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
    raw.pick_types(eeg=True, exclude=['ECG', 'EMG', 'EOG', 'SLI', 'ABD', 'ABDO', 'Status',
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
        orig_raw.plot(duration=15, n_channels = 20, title="Original " + filename, remove_dc = False)
        # ICA Components in Scalp
        ica.plot_components(title="ICA Components " + filename)
        # ICA Sources Time Series
        ica.plot_sources(orig_raw, title="ICA Sources " + filename)
        # Without Artifacts EEG Plot
        raw.plot(duration=15, n_channels = 20, title="Preprocessed " + filename, remove_dc = False)
        # Epochs Plot
        epochs.plot(n_epochs=10, n_channels= 20, title="EEG 2s Epochs " + filename)     
    
    return raw, ica

#%% Removes noisy epochs
def clean_epochs(filename, epochs, plot=False):
    ar = AutoReject(verbose=False, random_state=42)
    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
    
    if plot == True:
        reject_log.plot_epochs(epochs, title="Clean Epochs "  + filename)
    return epochs_clean, reject_log

#%% Get bandpowers from Epochs

def epochs_selection_bandpower(epochs, allVars=False):
    # bands names
    bd_names = ['Delta', 'Theta', 'Alpha', 'Beta']
    
    # bands ranges
    bands = [(1, 4, 'Delta'), (4, 8, 'Theta'),
             (8, 12, 'Alpha'), (12, 30, 'Beta')]
    # channel names
    ch = epochs.ch_names
    
    n_epochs = np.shape(epochs._data)[0]
    
    # initialize arrays 
    ms = np.zeros((n_epochs, 4))
    ms_dist = np.zeros((n_epochs, 1))
    
    for i in range (n_epochs):
        # compute bandpowers
        bd = yasa.bandpower(data=epochs._data[i,:,:], sf=256, ch_names=ch,
                            hypno=None, relative=True, bands=bands)
        
        # compute means
        bd_means = bd.mean(axis=0)
        b = np.array([bd_means['Delta'], bd_means['Theta'], 
                      bd_means['Alpha'], bd_means['Beta']])
              
        # add means to array with all epochs
        ms[i,:] = b 
        
        # compute variance and power
        var = np.var(b)
        power = bd_means['TotalAbsPow']

        # selection measure
        measure = power / var * 10e9
        
        # add measure to array
        ms_dist[i,0] = measure
          
    
    # transform to DataFrame
    bd_powers = pd.DataFrame(ms, columns=bd_names)
    ms_dist = pd.DataFrame(ms_dist, columns=['Measure'])
    
    # select threshold for all bands
    th = int(np.rint(n_epochs / 4))
    
    # sort epochs by distribution measure
    ms_dist_sorted = ms_dist.sort_values(by='Measure', ascending=False)
    
    # get first N indexes
    idx_n = np.array(ms_dist_sorted[:th].index)
    
    # create new object with selected epochs
    s_epoch = epochs.copy()
    s_epoch._data = epochs._data[idx_n,:,:]
    s_epochs = [s_epoch]
    
    # only 0.25 * n_epochs highest ranked epochs are selected
    N = 0.25
    th = int(np.rint(N * n_epochs))

    # select N highest ranked values for each band 
    idxs = []
    min_powers = []
    
    for bd_n in bd_names:
        # sort epochs by power
        bd_n_sorted = bd_powers.sort_values(by=bd_n, ascending=False)
        # get indexes of highest th epochs
        idx_n = np.array(bd_n_sorted[:th].index)
        idxs.append(idx_n)
        
        # save minimum power in selected epochs
        min_power = bd_n_sorted[bd_n].iloc[[th-1]].values.item()
        min_powers.append(min_power)
        
        # create new epochs object, with selected ones
        s_epoch = epochs.copy()
        s_epoch._data = epochs._data[idx_n,:,:]
        s_epochs.append(s_epoch)
        
    # conversion to array    
    min_powers = np.array(min_powers)
    
    # insert "global" in bands names, for the record
    bd_names.insert(0, 'Global')
    
    if allVars == True:
        return bd_names, th, ms_dist_sorted, idxs, min_powers, s_epochs
    else:
        return bd_names, s_epochs

   
#%% Run
# filenames = pd.read_excel('Metadata_train.xlsx')['Filename']

# for filename in filenames[[1]]:
#     saved_epochs = getPickleFile('../PreProcessed_Data/' + filename)
#     bd_names, s_epochs = epochs_selection_bandpower(saved_epochs)

