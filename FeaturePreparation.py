from Pickle import getPickleFile, createPickleFile
import pandas as pd
import numpy as np

#%% Auxiliary functions

# retrieves all saved features (conn + graphs)
def _get_saved_features():
    IMCOH = getPickleFile('../Features/' + 'imcoh')
    PLV = getPickleFile('../Features/' + 'plv')
    MI = getPickleFile('../Features/' + 'mi')
    PDC = getPickleFile('../Features/' + 'pdc')
    graph_ms = getPickleFile('../Features/' + 'graphMeasures')
    
    return {'imcoh': IMCOH, 'plv': PLV, 'mi': MI, 'pdc': PDC}, graph_ms


# transforms connectivity array into flat DataFrame
def _triangular_to_1D(ft_arr, ch_names, filename, bd_name, ms_type, ms_name):
    
    bd_n = bd_name
    ms_t = ms_type[0]
    labels = []
    for ch_n in ch_names:
        for ch_i in ch_names:
            labels.append(ms_name + '-' + bd_n + ms_t + '-' + ch_n + '-' + ch_i )
     
    ft_flat = np.reshape(ft_arr, newshape=(1,len(labels)), order='F')
    ft_df = pd.DataFrame(data=ft_flat, index=[filename], columns=labels)
    return ft_df.loc[:, (ft_df != 0).any(axis=0)]

# transforms graph measures to flat DataFrame
def _graph_measures_to_1D(gr_ms, ch_names, filename, ms_conn, bd_name, gr_name):
    
    labels = []
    templ_str = gr_name + '-' + ms_conn + '-' + bd_name
    if gr_name == 'global_efficiency':
        labels.append(templ_str)
    else:
        for gm, cn in zip(gr_ms, ch_names):
            labels.append(templ_str + '-' + cn)
    
    gr_flat = np.reshape(gr_ms, newshape=(1,len(labels)), order='F')
    gr_df = pd.DataFrame(data=gr_flat, index=[filename], columns=labels)
    
    return gr_df
    
#%% Produce Features Array for ML

def make_features_array(conn_ms, graph_ms):
        
    ch_names = ['Fz', 'Cz', 'Pz', 'Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5', 
                          'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6']
    
    filenames = pd.read_excel('Metadata_train.xlsx')['Filename']
    ms_conns = ['imcoh', 'plv', 'mi', 'pdc']
    ms_types = ['Mean', 'Std']
        
    allFeatures = pd.DataFrame()
    
    for filename in filenames[[0]]:
        features_row = pd.DataFrame()
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
                for ms_type in ms_types:
                    ms_c = conn_ms[ms_conn][filename][bd_name][ms_type]
                    ft_df = _triangular_to_1D(ms_c, ch_names, filename, \
                                              bd_name, ms_type, ms_conn)
                    features_row = pd.concat([features_row, ft_df], axis=1)
                
                for gr_name in gr_names:
                    gr_ms = graph_ms[filename][ms_conn][bd_name][gr_name]
                    gr_df = _graph_measures_to_1D(gr_ms, ch_names, filename, \
                                                  ms_conn, bd_name, gr_name)
                    features_row = pd.concat([features_row, gr_df], axis=1)
                
        allFeatures = pd.concat([allFeatures, features_row], axis=0)
    
    return allFeatures


#%% Run

conn_ms, graph_ms = _get_saved_features()
allFeatures = make_features_array(conn_ms, graph_ms)
# createPickleFile(allFeatures, '../Features/' + 'allFeatures')








