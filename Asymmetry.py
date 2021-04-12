import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from FeatureExtraction import band_power_measures
from Pickle import getPickleFile, createPickleFile
import brainconn
from Pickle import createPickleFile, getPickleFile
from DataPreparation import get_saved_features
            

#%% Auxiliary Function

# Compute ratio between subgroups
def _compute_ratio(a, b):

    max_val = max(a, b)
    min_val = min(a, b)
    
    if min_val == 0:
        return 0
    else:
        return max_val / min_val


#%% Computation of Asymmetry Measures in pairs

def compute_asymmetry_measures(graph_ms):
    
    filenames = pd.read_excel('Metadata_train.xlsx')['Filename']    
    
    pairs = [['FR', 'FL'], ['BR', 'BL']]
    
    asy_ms = {}
    
    for filename in filenames[0:5]:
        
        allPairs = pd.DataFrame()
        
        for pair in pairs:
            # get this subjects' graph measures
            gr_df = graph_ms[filename]
            
            # get features' of interest
            fts_names = gr_df.index.to_list()
            sub1_names = [i for i in fts_names if i.endswith(pair[0]) or i.endswith(pair[0] + '-Mean') ]
            sub2_names = [i for i in fts_names if i.endswith(pair[1]) or i.endswith(pair[1] + '-Mean') ]
            
            # get this pair's features 
            sub1_fts = gr_df.copy().filter(items=sub1_names, axis=0).values
            sub2_fts = gr_df.copy().filter(items=sub2_names, axis=0).values
            
            # build new ratio names
            ratios_names = [i[:-5] if i.endswith('-Mean') else i for i in sub1_names]
            ratios_names = [i + 'vs' + pair[1] for i in ratios_names]
            
            # compute asymmetry ratios
            ratios_vals = [_compute_ratio(s1, s2) for s1, s2 in zip(sub1_fts, sub2_fts)]
            ratios_vals = [float(i) for i in ratios_vals]
            
            # build this pair's dataframe
            pair_df = pd.DataFrame(data=ratios_vals, columns=[filename], index=ratios_names)
            
            # add this pair to all pairs dataframe
            allPairs = pd.concat([allPairs, pair_df], axis=0)
    
        asy_ms[filename] = allPairs
        
    return asy_ms
