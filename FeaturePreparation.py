from Pickle import getPickleFile, createPickleFile
import pandas as pd
import numpy as np

#%% Auxiliary functions

# transforms connectivity array into flat DataFrame
def _triangular_to_1D(ft_arr, filename, bd_name, ms_type, ms_name):
    ch_names = ['Fz', 'Cz', 'Pz', 'Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5', 
                          'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6']
    bd_n = bd_name
    ms_t = ms_type[0]
    labels = []
    for ch_n in ch_names:
        for ch_i in ch_names:
            labels.append(ms_name + '-' + bd_n + ms_t + '-' + ch_n + '-' + ch_i )
     
    ft_flat = np.reshape(ft_arr, newshape=(1,len(labels)), order='F')
    ft_df = pd.DataFrame(data=ft_flat, index=[filename], columns=labels)
    return ft_df.loc[:, (ft_df != 0).any(axis=0)]

#%% Run

IMCOH = getPickleFile('../Features/' + 'imcoh')
PLV = getPickleFile('../Features/' + 'plv')
MI = getPickleFile('../Features/' + 'mi')
PDC = getPickleFile('../Features/' + 'pdc')

fts = {'imcoh': IMCOH, 'plv': PLV, 'mi': MI, 'pdc': PDC}

filenames = pd.read_excel('Metadata_train.xlsx')['Filename']
ms_names = ['imcoh', 'plv', 'mi', 'pdc']
bd_names = ['Global', 'Delta', 'Theta', 'Alpha', 'Beta']
ms_types = ['Mean', 'Std']

imcoh = {}
plv = {}
mi = {}
pdc = {}

allFeatures = pd.DataFrame()

for filename in filenames:
    features_row = pd.DataFrame()
    for ms_name in ms_names:
        if ms_name == 'mi':
            bd_names = ['Global']
        else:
            bd_names = ['Global', 'Delta', 'Theta', 'Alpha', 'Beta']
            
        for bd_name in bd_names:
            for ms_type in ms_types:
                ft = fts[ms_name][filename][bd_name][ms_type]
                ft_df = _triangular_to_1D(ft, filename, bd_name, ms_type, ms_name)
                features_row = pd.concat([features_row, ft_df], axis=1)
                
    allFeatures = pd.concat([allFeatures, features_row], axis=0)

createPickleFile(allFeatures, '../Features/' + 'allFeatures')






