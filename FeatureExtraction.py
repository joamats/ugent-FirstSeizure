import mne
import pandas as pd
import numpy as np
import scot
from scot.eegtopo import eegpos3d
from sklearn.feature_selection import mutual_info_regression
from BandpowerCorrection import bandpower_1f_correction
from spectral_connectivity import Multitaper, Connectivity

#%% Auxiliary functions

# transforms 2D matrix to 3D
def _3D_to_triangular(m_3D):
    
    n = np.shape(m_3D)[0]
    m_2D = np.zeros((n,n))
    
    for i in range(n):
        m_2D[i,:] = np.matrix.transpose(m_3D[i,:,:])
    
    return m_2D

# computes mean and standard deviation for the feature data matrix
def _compute_feature_mean(feature_data):
    
    return np.mean(feature_data, axis=2)

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
def _map_bins_to_indices(band, fs=128, bins_fft=256, toolbox='scot'):
    
    if toolbox == 'scot':
        f_step = 0.5 * fs / bins_fft
    elif toolbox == 'eden-kramer':
        f_step = 0.4
        
    limits = [int(b / f_step) for b in band]
    
    return range(limits[0], limits[1] + 1)

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
            # Combine all epochs with MI median and std 
            mutual_infos[channel_1][channel_2][0] = np.median(all_mi)
            std[channel_1][channel_2][0] = np.std(all_mi)
    
    # transforms the 3D matrix in 2D    
    
    return _3D_to_triangular(mutual_infos)

#%% Partial Directed Coherence

def partial_directed_coherence(epochs, plot=False, toolbox='scot'):
    
    if toolbox == 'scot':
        # get number of channels
        n_channels = len(epochs.ch_names)
        
        # get electrodes coordinates
        locs = _get_scot_locations(epochs)
            
        # multivariate VAR
        var = scot.var.VAR(model_order=8)
        var.fit(epochs._data)
        
        # workspace settings
        fs = 128
        bins_fft = 256
        
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
            ws.plot_outside_topo = True
            fig = ws.plot_connectivity_topos()    
            # compute connectivity
            pdc, fig = ws.get_connectivity('PDC', plot=fig)
            ws.show_plots()
        else:
             pdc = ws.get_connectivity('PDC', plot=False)
        
        return pdc
    
    elif toolbox == 'eden-kramer':
        # swap axes
        time_series = np.swapaxes(epochs._data, axis1=0, axis2=2)
        time_series = np.swapaxes(time_series, axis1=1, axis2=2)
        
        m = Multitaper(time_series,
               sampling_frequency=128,
               time_halfbandwidth_product=2,
               start_time=0)

        c = Connectivity(fourier_coefficients=m.fft(),
                         frequencies=m.frequencies,
                         time=m.time)
        
        pdc = c.partial_directed_coherence()[0,:,:,:]
        
        pdc = np.swapaxes(pdc, axis1=0, axis2=2)
        pdc = np.swapaxes(pdc, axis1=0, axis2=1)
        
        return pdc
    
    
#%% Connectivity Features Extractor

def extract_features(bd_names, epochs):
    
    bands = {'Global': [2,30], 'Delta': [2, 4], 'Theta': [4, 8],
         'Alpha': [8,12], 'Beta': [12, 30]}
    
    imcohs = {}
    plvs = {}
    pdcs = {}
    
    for bd_n in bd_names:
        f_min, f_max = bands[bd_n]
        
        # IMCOH
        imcoh = mne.connectivity.spectral_connectivity(epochs[bd_n], method = "imcoh", 
                                  sfreq=128, fmin=f_min, fmax=f_max, 
                                  faverage=False, verbose=False)

        imcohs[bd_n] = _compute_feature_mean(imcoh[0])
           
        # PLV
        plv = mne.connectivity.spectral_connectivity(epochs[bd_n], method = "plv", 
                                  sfreq = 128, fmin=f_min, fmax=f_max,
                                  faverage=False, verbose=False)   

        plvs[bd_n] = _compute_feature_mean(plv[0])

        # MI (only for Global band)
        if(bd_n == 'Global'):
            mi = {'Global': mutual_information(epochs[bd_n])}
               
        # PDC
        idxs_bd = _map_bins_to_indices(bands[bd_n], toolbox='scot')
        pdc = partial_directed_coherence(epochs[bd_n], plot=False, toolbox='scot')
        pdcs[bd_n] = _compute_feature_mean(pdc[:,:,idxs_bd])
                
    return imcohs, plvs, mi, pdcs

#%% Subrgroups' connectivity features 

# Filter of connectivity matrix
def _features_subgroup_combination(conn, subgroup, conn_name):
    
    ch_names = ['Fz', 'Cz', 'Pz', 'Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3',
                'T5', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6']
    
    if conn_name != 'pdc':
        conn = conn + conn.T - np.diag(np.diag(conn))
    
    if conn_name == 'imcoh':
        conn = abs(conn)
        
    conn_df = pd.DataFrame(data=conn, index=ch_names, columns=ch_names)
    return conn_df.filter(items=subgroup, axis=1).filter(items=subgroup, axis=0)
    
# Compute connectivity mean and std
def _conn_mean_std(conn_df, filename, conn_name, bd_name, sub_name):
    
    # if pdc, simply fill diagonal with zeros
    if conn_name == 'pdc':
        tr = np.copy(conn_df)
        np.fill_diagonal(tr,0)
    # if not pdc, get the lower triangular matrix
    else:
        tr = np.tril(conn_df)   
        
    m = np.mean(tr[tr!=0])
    s = np.std(tr[tr!=0])
    
    m_name = conn_name + '-' + bd_name + '-' + sub_name + '-Mean'
    s_name = conn_name + '-' + bd_name + '-' + sub_name + '-Std'
    
    return pd.DataFrame(data=[m,s], index=[m_name, s_name], columns=[filename])

# Final Computation for subgroups
def compute_connectivity_measures(fts):
    filenames = pd.read_excel('Metadata_train.xlsx')['Filename']
    
    subgroups = {   'FR': ['Fp1', 'F7', 'T3', 'F3', 'C3', 'Fz', 'Cz'],
                    'FL': ['Fp2', 'F8', 'T4', 'F4', 'C4', 'Fz', 'Cz'],
                    'BR': ['T3', 'T5', 'O1', 'C3', 'P3', 'Cz', 'Pz'],
                    'BL': ['T4', 'T6', 'O2', 'C4', 'P4', 'Cz', 'Pz'] }
    
    subgroups_names = ['FR', 'FL', 'BR', 'BL']
    
    conn_names = ['imcoh', 'plv', 'mi', 'pdc']
    
    conn_ms = {}
    
    for filename in filenames:
        df_all = pd.DataFrame()
        for conn_n in conn_names:
            if conn_n == 'mi':
                bd_names = ['Global']
            else:                           
                bd_names = ['Global', 'Delta', 'Theta', 'Alpha', 'Beta']
                
            for bd_n in bd_names:
                ft = fts[conn_n][filename][bd_n]
    
                for sub_n in subgroups_names:
                    chs = subgroups[sub_n]
                    ft_comb = _features_subgroup_combination(ft, chs, conn_n)
                    df_single = _conn_mean_std(ft_comb, filename, conn_n, bd_n, sub_n)
                    df_all = pd.concat([df_all, df_single], axis=0)
                    
        conn_ms[filename] = df_all
    
    return conn_ms 
                

#%% Bandpower
def band_power_measures(epochs, sub_name, filename):
        
    bands = [(2, 4, 'Delta'), (4, 8, 'Theta'),
             (8, 12, 'Alpha'), (12, 30, 'Beta')]
    
    bd_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'TotalAbsPow']

    bd_power_measures = pd.DataFrame()
    
    for bd_n in bd_names:
        bd_means = []
                
        # Total Absolute Power is computed for global band
        bd_na = 'Global' if bd_n == 'TotalAbsPow' else bd_n
        
        # channel names
        ch = epochs[bd_na].ch_names
        
        # number of epochs
        n_epochs = np.shape(epochs[bd_na]._data)[0]
        
        for i in range (n_epochs):
            # compute bandpowers
            bd, _, _ = bandpower_1f_correction(data=epochs[bd_na]._data[i,:,:], 
                                               sf=128, ch_names=ch,
                                               hypno=None, relative=True,
                                               bands=bands)
            
            # combination in subgroup's channels
            bd_means.append(bd[bd_n].mean(axis=0))
    
        # combination over all epochs
        m = np.median(bd_means)
        s = np.std(bd_means)
        
        # naming
        m_name = 'bdp-' + bd_n + '-' + sub_name + '-Median'
        s_name = 'bdp-' + bd_n + '-' + sub_name + '-Std'
        
        bd_power_band = pd.DataFrame(data=[m,s], columns=[filename], index=[m_name, s_name])
        bd_power_measures = pd.concat([bd_power_measures, bd_power_band], axis=0)
                                     
    return bd_power_measures


#%% Subgroups' Bandpowers

# Drops channels, calculates band powers 
def _band_powers_subgroup(saved_epochs, chs_subgroup, sub_name, filename):
    
    s_epochs = {}
    
    for key in saved_epochs:

        ch_names = saved_epochs[key].ch_names
        chs_to_drop = [i for i in ch_names if i not in chs_subgroup]    
        epochs_use = saved_epochs[key].copy().drop_channels(chs_to_drop) 
        s_epochs[key] = epochs_use
        
    bd_powers = band_power_measures(s_epochs, sub_name, filename)
   
    return bd_powers

def extract_bandpowers(epochs, filename):
    
    subgroups = {
        'FR': ['Fp1', 'F7', 'T3', 'F3', 'C3', 'Fz', 'Cz'],
        'FL': ['Fp2', 'F8', 'T4', 'F4', 'C4', 'Fz', 'Cz'],
        'BR': ['T3', 'T5', 'O1', 'C3', 'P3', 'Cz', 'Pz'],
        'BL': ['T4', 'T6', 'O2', 'C4', 'P4', 'Cz', 'Pz'] }
        # 'R': ['Fz', 'Cz', 'Pz', 'Fp1', 'F7', 'F3', 'T3', 'C3', 'T5', 'P3', 'O1'],
        # 'L': ['Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'F8', 'C4', 'T4', 'P4', 'T6', 'O2'],
        # 'ALL': epochs.ch_names }
       
    subgroups_names = ['FR', 'FL', 'BR', 'BL'] #, 'R', 'L', 'ALL']
    
    df_all = pd.DataFrame()
    
    for sub_n in subgroups_names:
        chs = subgroups[sub_n]
        df_single = _band_powers_subgroup(epochs, chs, sub_n, filename)
        df_all = pd.concat([df_all, df_single], axis=0)
        
    return df_all
