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

#%%
def filter_eeg(raw):
    f_stop = 50
    f_low = 1
    f_high = 100
    
    raw.notch_filter(f_stop)
    raw.filter(f_low, f_high)
    return raw

#%% Load Filenames
filenames = pd.read_excel('Metadata_train.xlsx')['Filename']

#%%
raws = list()
icas = list()
idxs = list()
li = 0

for subj in filenames.index:
    # import signal
    raw = import_eeg(filenames[subj])
    
    # filter signal
    raw = filter_eeg(raw)

    # fit ICA
    ica = mne.preprocessing.ICA(n_components=7, random_state=42)
    ica.fit(raw, verbose=False)
    
    # ica.plot_components()
    # ica.plot_sources(raw)
    
    if(subj <= 1):
        icas.append(ica)
        raws.append(raw)
    else:
        icas[1] = ica
        raws[1] = raw
        
    mne.preprocessing.corrmap(icas, template=(0, 0), threshold=0.8, label='blink', plot = False, verbose = False)
        
    idx = [icas[li].labels_['blink']]
    idxs.append(idx)
    
    
    # aqui verificar se o idx é vazio ou tem mais que 1 ICA component
    # se sim, meter os plots, senão avançar!
    
    
    ica.exclude = [item for sublist in idx for item in sublist]
    orig_raw = raw.copy()
    ica.apply(raw, verbose=False)
    
    # orig_raw.plot(duration=10, n_channels = 19, title="Original", remove_dc = False)
    #raw.plot(duration=10, n_channels = 19, title="After ICA", remove_dc = False)
    
    li=1
    