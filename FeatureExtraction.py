import mne
import numpy as np
import scot
from scot.eegtopo import eegpos3d
from sklearn.feature_selection import mutual_info_regression

#%% Mutual Information

def mutual_information(epochs):
    
    mutual_infos = np.zeros((np.shape(epochs._data)[1], np.shape(epochs._data)[1], 1))
    std = np.zeros((np.shape(epochs._data)[1], np.shape(epochs._data)[1], 1))
    
    # computes for each channel combination the averaged MI
    for channel_1 in range (1, np.shape(epochs._data)[1]):
        for channel_2 in range (channel_1):
            all_mi = []
            for singleEpoch in range (np.shape(epochs._data)[0]):
                x = epochs._data[singleEpoch][channel_1]
                x = x.reshape(-1,1)
                y = epochs._data[singleEpoch][channel_2]
                mi = mutual_info_regression(x, y, random_state=42)
                all_mi.append(mi)
            #Saves the all the MI mean and std in the right position
            mutual_infos[channel_1][channel_2][0] = np.mean(all_mi)
            std[channel_1][channel_2][0] = np.std(all_mi)
    
    m = {}
    
    # transforms the 3D matrix in 2D    
    m['Mean'] = _3D_to_triangular(mutual_infos)
    m['Std'] = _3D_to_triangular(std)
        
    return m

#%% Partial Directed Coherence

def partial_directed_coherence(epochs, plot=False, band=[]):
    # get number of channels
    n_channels = len(epochs.ch_names)
    
    # get electrodes coordinates
    locs = _get_scot_locations(epochs)
        
    # multivariate VAR
    var = scot.var.VAR(model_order=8)
    var.fit(epochs._data)
    
    # workspace settings
    fs = 256
    bins_fft = 512
    
    # SCoT workspace
    ws = scot.Workspace(var, reducedim='no_pca', fs=fs,
                        nfft=bins_fft, locations=locs)
    ws.set_data(epochs._data)
    
    # manually set mixing matrices to identity - we want no mixing
    ws.set_premixing(np.eye(n_channels))
    ws.mixing_ = np.eye(n_channels)
    
    # manually set activations equal to data - each source is a channel
    ws.activations_ = epochs._data
    
    # fit var
    ws.fit_var()
    
    # plotting settings
    if plot:
        ws.plot_f_range = band
        ws.plot_outside_topo = True
        fig = ws.plot_connectivity_topos()    
        # compute connectivity
        pdc, fig = ws.get_connectivity('PDC', plot=fig)
        ws.show_plots()
    else:
         pdc = ws.get_connectivity('PDC', plot=False)
    
    return pdc
    

#%% Auxiliary functions

# transforms 2D matrix to 3D
def _3D_to_triangular(m_3D):
    n = np.shape(m_3D)[0]
    m_2D = np.zeros((n,n))
    
    for i in range(n):
        m_2D[i,:] = np.matrix.transpose(m_3D[i,:,:])
    
    return m_2D

# computes mean and standard deviation for the feature data matrix
def _compute_feature_mean_std(feature_data):
    feature_data_mean_std = {}
    feature_data_mean_std['Mean'] = np.mean(feature_data, axis=2)
    feature_data_mean_std['Std'] = np.std(feature_data, axis=2)
    
    return feature_data_mean_std

# get electrode location
def _get_scot_locations(epochs):
        # construct positions struct
        pos = eegpos3d.construct_1020_easycap(variant=0)
        # channels renamed as {'T7': 'T3', 'P7': 'T5', 'T8': 'T4', 'P8': 'T6'}
        ch_names = ['Fz', 'Cz', 'Pz', 'Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T7', 
                   'P7', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T8', 'P8']
        
        locs = [pos[i] for i in ch_names]
        
        n_channels = len(epochs.ch_names)
        locations = np.zeros((n_channels, 3))
        
        for i in range(n_channels):
            locations[i,0] = locs[i].list[0]
            locations[i,1] = locs[i].list[1]
            locations[i,2] = locs[i].list[2]
        
        return locations

# mapping of fft bins to subscriptable indices 
def _map_bins_to_indices(band, fs=256, bins_fft=512):
    f_step = 0.5 * fs / bins_fft
    limits = [int(b / f_step) for b in band]
    return range(limits[0], limits[1] + 1)


#%% Final Feature Extractor

def extract_features(bd_names, epochs):
    
    bands = {'Global': [2.5,30], 'Delta': [2.5, 4], 'Theta': [4, 8],
         'Alpha': [8,12], 'Beta': [12, 30]}
    
    imcohs = {}
    plvs = {}
    pdcs = {}
    
    for bd_n in bd_names:
        f_min, f_max = bands[bd_n]
        
        # IMCOH
        imcoh = mne.connectivity.spectral_connectivity(epochs[bd_n], method = "imcoh", 
                                  sfreq = 256, fmin=f_min, fmax=f_max, 
                                  faverage=False, verbose = False)

        imcohs[bd_n] = _compute_feature_mean_std(imcoh[0])
           
        # PLV
        plv = mne.connectivity.spectral_connectivity(epochs[bd_n], method = "plv", 
                                  sfreq = 256, fmin=f_min, fmax=f_max,
                                  faverage=False, verbose = False)   

        plvs[bd_n] = _compute_feature_mean_std(plv[0])

        # MI (only for Global band)
        if(bd_n == 'Global'):
            mi = {'Global': mutual_information(epochs[bd_n])}
               
        # PDC
        idxs_bd = _map_bins_to_indices(bands[bd_n])
        pdc = partial_directed_coherence(epochs[bd_n], plot=False)
        pdcs[bd_n] = _compute_feature_mean_std(pdc[:,:,idxs_bd])
        
    return imcohs, plvs, mi, pdcs
                