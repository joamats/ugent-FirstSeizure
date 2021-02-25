import mne
import numpy as np
import pandas as pd


def import_eeg(filename):
    raw = mne.io.read_raw_edf('../Dataset/{}_0000.Export.edf'
                              .format(filename), preload=True, verbose=False)
    # Rename channels (if necessary)
    if 'FP1' in raw.info['ch_names']:
        raw.rename_channels({'FP1': 'Fp1', 'FP2': 'Fp2'})
    if 'T7' in raw.info['ch_names']:
        raw.rename_channels({'T7': 'T3', 'P7': 'T5', 'T8': 'T4', 'P8': 'T6'})
    # Select relevant channels
    raw.pick_types(eeg=True, exclude=['ECG', 'EMG', 'EOG', 'SLI', 'ABD', 'ABDO', 'Status',
                                      'T1', 'T2', 'CP1', 'CP2', 'CP5', 'CP6', 'FC1', 'FC2', 'FC5', 'FC6'])
    # Average referencing
    raw.set_eeg_reference()
    # Set 10-20 montage
    ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(ten_twenty_montage)
    return raw

filenames = pd.read_excel('Metadata_train.xlsx')['Filename']
f = filenames[3]
raw = import_eeg(f)

f_stop = np.arange(50, 128, 50)
f_low = 1
f_high = 100
raw.notch_filter(f_stop)
raw.filter(f_low, f_high)

print(raw.info)

# set up and fit the ICA
ica = mne.preprocessing.ICA(n_components=15, random_state=43)
ica.fit(raw)
ica.plot_components()

# blinks artifacts
ica.exclude = [0]  # exclude EOG artifacts

# to-do: ECG artifacts

orig_raw = raw.copy()
ica.apply(raw)

orig_raw.plot(duration=10, n_channels = 19, title="Original", remove_dc=False)
raw.plot(duration=10, n_channels = 19, title="After ICA", remove_dc=False)

# raws = list()
# icas = list()

# for subj in range(4):
#     raw = import_eeg(filenames[subj])
#     # fit ICA
#     ica = mne.preprocessing.ICA(n_components=15, random_state=43)
#     ica.fit(raw)
#     raws.append(raw)
#     icas.append(ica)

# mne.preprocessing.corrmap(icas, template=(0, 0), threshold=0.9, label='blink')

# for subj in range(4):
#     icas[subj].plot_overlay(raws[subj], exclude=icas[subj].labels_['blink'],picks='eeg')
