from Pickle import getPickleFile
import seaborn as sb
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#%% 
def _find_feature_indices(fts_names, conn_ms, bd_name, graph_ms):
    
    t_str = graph_ms + '-' + conn_ms + '-' + bd_name
    
    idxs = []
    for i, ft_n in enumerate(fts_names):
        r = ft_n.find(t_str)
        
        if r != -1:
            idxs.append(i)
            
    return idxs, t_str


def _get_features_mean(data, idxs):
    
    return np.mean(data[:,idxs], axis=1)

def _get_features_max(data, idxs):
    
    return np.max(data[:, idxs], axis=1)
    
#%%
    
fts_names = getPickleFile('../3_ML_Data/' + 'featuresNames')
datasets = getPickleFile('../3_ML_Data/' + 'datasets')

data = datasets[0]['train'][0]['X_tr']
labels = pd.DataFrame(datasets[0]['train'][0]['y_tr'], columns=['labels'])

ms_conns = ['imcoh', 'plv', 'mi', 'pdc']

fts = {}
fts_df = pd.DataFrame()
 
for ms_conn in ms_conns:
    if ms_conn == 'mi':
        bd_names = ['Global']
        
    elif ms_conn == 'pdc':
        gr_names = ['betweness_centrality', 'clustering_coefficient',
                'global_efficiency', 'incoming_flow', 'outgoing_flow']
        bd_names = ['Global', 'Delta', 'Theta', 'Alpha', 'Beta']
    else:
        gr_names = ['betweness_centrality', 'clustering_coefficient',
                'global_efficiency', 'node_strengths']
        
        bd_names = ['Global', 'Delta', 'Theta', 'Alpha', 'Beta']
        
    for bd_name in bd_names:

        ft_gr = pd.DataFrame()
        for gr_name in gr_names:
            
            idxs, t_str = _find_feature_indices(fts_names, \
                             ms_conn, bd_name, gr_name)
            
            m = pd.DataFrame(data=_get_features_mean(data, idxs), columns=[t_str])
            maximum=pd.DataFrame(data=_get_features_max(data, idxs), columns=[t_str])
            # ft_gr = pd.concat([ft_gr, m], axis=1)
            ft_gr = pd.concat([ft_gr, maximum], axis=1)
            
        ft_gr = pd.concat([ft_gr, labels], axis=1)
                
        sb.pairplot(ft_gr, hue='labels', diag_kind='kde')
        plt.suptitle('Max' + ms_conn + '-' + bd_name, y=1.05)

