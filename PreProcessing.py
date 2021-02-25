import mne
import numpy as np
import pandas as pd

#%%
def import_eeg(filename):
    raw = mne.io.read_raw_edf('../Dataset/{}_0000.Export.edf'
                              .format(filename), preload=True, verbose=False)
    # Rename channels (if necessary)
    if 'FP1' in raw.info['ch_names']:
        raw.rename_channels({'FP1': 'Fp1', 'FP2': 'Fp2'})
    if 'T7' in raw.info['ch_names']:
        raw.rename_channels({'T7': 'T3', 'P7': 'T5', 'T8': 'T4', 'P8': 'T6'})

    # Select relevant channels: EEG and ECG
    raw.pick_types(eeg=True, ecg=True, exclude=['EMG', 'EOG', 'SLI', 'ABD', 'ABDO', 'Status',
                                      'T1', 'T2', 'CP1', 'CP2', 'CP5', 'CP6', 'FC1', 'FC2', 'FC5', 'FC6'])
    
    # Average referencing
    raw.set_eeg_reference()
    # Add ECG Channel
    raw.set_channel_types({'ECG': 'ecg'})
    # Set 10-20 montage
    ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(ten_twenty_montage)
    return raw

#%% Load Data
filenames = pd.read_excel('Metadata_train.xlsx')['Filename']
f = filenames[0]
raw = import_eeg(f)

#%% Filter EEG
f_stop = 50
f_low = 1
f_high = 100

raw.notch_filter(f_stop)
raw.filter(f_low, f_high)
raw.plot_psd()
print(raw.info)

#%% ICA
ica = mne.preprocessing.ICA(n_components=7, random_state=43)
ica.fit(raw)
ica.plot_components()
ica.plot_sources(raw)

#%% Blinks artifacts
ica.plot_overlay(raw, exclude=[0], picks='eeg')
ica.plot_overlay(raw, exclude=[1], picks='eeg')

#%% ECG artifacts
ica.plot_overlay(raw, exclude=[2], picks='eeg')

#%% Exclude
ica.exclude = [0, 1, 3]

#%% Plot
orig_raw = raw.copy()
ica.apply(raw)

orig_raw.plot(duration=10, n_channels = 20, title="Original", remove_dc=False)
raw.plot(duration=10, n_channels = 20, title="After ICA", remove_dc=False)




