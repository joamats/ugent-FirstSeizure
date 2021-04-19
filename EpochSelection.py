import numpy as np
import pandas as pd
from BandpowerCorrection import bandpower_1f_correction

def epochs_selection_bandpower(epochs, allVars=False):
    # bands names
    bd_names = ['Delta', 'Theta', 'Alpha', 'Beta']
    
    # bands ranges
    bands = [(2, 4, 'Delta'), (4, 8, 'Theta'),
             (8, 12, 'Alpha'), (12, 30, 'Beta')]
    # channel names
    ch = epochs.ch_names
    
    # number of epochs
    n_epochs = np.shape(epochs._data)[0]
    
    # only 50 highest ranked epochs are selected
    th = 50
    
    # initialize arrays 
    ms = np.zeros((n_epochs, 4))
    ms_dist = np.zeros((n_epochs, 1))
    
    for i in range (n_epochs):
        # compute bandpowers
        bd, _, _ = bandpower_1f_correction(data=epochs._data[i,:,:], sf=128, ch_names=ch,
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
    
    # sort epochs by distribution measure
    ms_dist_sorted = ms_dist.sort_values(by='Measure', ascending=False)
    
    # get first N indexes
    idx_n = np.array(ms_dist_sorted[:th].index)
    
    # create new object with selected epochs
    s_epoch = epochs.copy()
    s_epoch._data = epochs._data[idx_n,:,:]
    s_epochs = {'Global': s_epoch}
    
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
        s_epochs[bd_n] = s_epoch
        
    # conversion to array    
    min_powers = np.array(min_powers)
    
    # insert "global" in bands names, for the record
    bd_names.insert(0, 'Global')
    
    if allVars == True:
        return bd_names, th, ms_dist_sorted, idxs, min_powers, s_epochs
    else:
        return bd_names, s_epochs