import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb
from Pickle import getPickleFile

#%% Plots the heat map for the inserted data_array, with the entered color bar label and band title
def plot_heatmap(data_array, bar_label, band_title):
    mask=np.ones((19,19))
    mask=np.triu(mask)
    
    ch_names_original = ['Fz', 'Cz', 'Pz', 'Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T7', 
                          'P7', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T8', 'P8']
    #New channel order
    ch_names_new = ['Fp1', 'Fp2', 'F8', 'F4', 'Fz', 'F3', 'F7', 'T7', 'C3', 'Cz',
                    'C4', 'T8', 'P8', 'P4', 'Pz', 'P3', 'P7', 'O1', 'O2']
    ch_indexes = []
    
    #New channel order's indexes on the ch_names_original variable
    for ch in ch_names_new:
        ch_indexes.append(ch_names_original.index(ch))
    
    if np.allclose(data_array, np.tril(data_array)): #Only for triangular matrices
        #makes the triangular matrix into a simetric matrix
        data_array=data_array+data_array.T-np.diag(np.diag(data_array))
        b=np.zeros((19,19))
        #Change the order of the channels
        b=data_array[:,ch_indexes]
        b=b[ch_indexes,:]
        data_array=np.tril(b)
        fig=sb.heatmap(data_array, cmap='viridis',
                    mask=mask, cbar_kws={'label': bar_label},
                    xticklabels=ch_names_new, yticklabels=ch_names_new)
    else:
        b=np.zeros((19,19))
        #Change the order of the channels
        b=data_array[:,ch_indexes]
        b=b[ch_indexes,:]
        data_array = b
        fig=sb.heatmap(data_array, cmap='viridis',
                    cbar_kws={'label': bar_label},
                    xticklabels=ch_names_new, yticklabels=ch_names_new)
    plt.xlabel("Channel 2")
    plt.ylabel("Channel 1")
    plt.title(band_title)
    
    
    return fig

#%%
filenames = pd.read_excel('Metadata_train.xlsx')['Filename']

imcoh = getPickleFile('../2_Features_Data/' + 'imcoh')
mi = getPickleFile('../2_Features_Data/' + 'mi')
plv = getPickleFile('../2_Features_Data/' + 'plv')
pdc = getPickleFile('../2_Features_Data/' + 'pdc')

bd_names = ['Global'] #, 'Delta', 'Theta', 'Alpha', 'Beta']

for filename in filenames[[201]]:
    for bd in bd_names:
        plot_heatmap(imcoh[filename][bd]['Mean'], 'ImCoh', str(filename+'_'+bd))
        plt.figure()
        
    plot_heatmap(mi[filename]['Global']['Mean'], 'MI', str(filename+'_Global'))
    plt.figure()
    
    for bd in bd_names:
        plot_heatmap(plv[filename][bd]['Mean'], 'PLV', str(filename+'_'+bd))
        plt.figure()
        
    for bd in bd_names:
        plot_heatmap(pdc[filename][bd]['Mean'], 'PDC', str(filename+'_'+bd))
        plt.figure()
