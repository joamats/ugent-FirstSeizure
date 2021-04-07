from Pickle import getPickleFile, createPickleFile
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

#%% Auxiliary functions

# retrieves all saved features (conn + graphs)
def get_saved_features(withGraphs=False):
    bdp_ms = getPickleFile('../2_Features_Data/128Hz/' + 'bdp')
    asy_ms = getPickleFile('../2_Features_Data/128Hz/' + 'asymmetryMeasures')
    
    IMCOH = getPickleFile('../2_Features_Data/128Hz/' + 'imcoh')
    PLV = getPickleFile('../2_Features_Data/128Hz/' + 'plv')
    MI = getPickleFile('../2_Features_Data/128Hz/' + 'mi')
    PDC = getPickleFile('../2_Features_Data/128Hz/' + 'pdc')
    
    if withGraphs:
        graph_ms = getPickleFile('../2_Features_Data/128Hz/' + 'graphMeasures')
        return bdp_ms, {'imcoh': IMCOH, 'plv': PLV, 'mi': MI, 'pdc': PDC}, graph_ms, asy_ms
    
    else:
        return bdp_ms, {'imcoh': IMCOH, 'plv': PLV, 'mi': MI, 'pdc': PDC}

# transforms bandpowers array into flat DataFrame
def _bandpowers_to_1D(ft_arr, filename):
    bd_names = ['Delta', 'Theta', 'Alpha', 'Beta']
    ms_types = ['Max','Mean','Median','Min','Std']
    labels = []
    ft_flat = []
    
    for bd_n in bd_names:
        for ms_t in ms_types:
            labels.append('bdpALL' + '-' + bd_n + '-' + ms_t)
            ft_flat.append(ft_arr[bd_n][ms_t])
         
    ft_flat = np.reshape(ft_flat, newshape=(1,len(labels)), order='F')
    ft_df = pd.DataFrame(data=ft_flat, index=[filename], columns=labels)
    
    return ft_df

# transforms connectivity array into flat DataFrame
def _triangular_to_1D(ft_arr, ch_names, filename, bd_name, ms_type, ms_name):
    
    bd_n = bd_name
    ms_t = ms_type
    labels = []
    for ch_n in ch_names:
        for ch_i in ch_names:
            labels.append(ms_name + '-' + bd_n + '-' + ms_t + '-' + ch_n + '-' + ch_i )
     
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

def _rename_dataframe_columns(stats, ms_conn, bd_name):
    cols_orig = stats.columns
    cols_new = [i + '-' +  ms_conn + '-' + bd_name for i in cols_orig]
    
    return stats.rename(columns=dict(zip(cols_orig, cols_new)))
       
    
#%% 

# Produce Features Array for ML
def make_features_array(bdp_ms, conn_ms, asy_ms, graph_ms, std=True):
        
    ch_names = ['Fz', 'Cz', 'Pz', 'Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5', 
                          'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6']
    
    filenames = pd.read_excel('Metadata_train.xlsx')['Filename']
    ms_conns = ['imcoh', 'plv', 'mi', 'pdc']
    
    if std:
        ms_types = ['Mean', 'Std']
    else:
        ms_types = ['Mean']
        
    allFeatures = pd.DataFrame()
    
    for filename in filenames:
        features_row = pd.DataFrame()
        ft_df = _bandpowers_to_1D(bdp_ms[filename], filename)
        features_row = pd.concat([features_row, ft_df], axis=1)
        
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
                
                # asymmetric graphs
                a_ms = asy_ms[filename][ms_conn][bd_name]
                # convert columns names
                a_ms = _rename_dataframe_columns(a_ms, ms_conn, bd_name)
                # concatenate it to all features vector
                features_row = pd.concat([features_row, a_ms], axis=1)
                
                # global graphs stats
                gr_ms = graph_ms[filename][ms_conn][bd_name]
                st_df = gr_ms['stats']
                st_df = _rename_dataframe_columns(st_df, ms_conn, bd_name)
                features_row = pd.concat([features_row, st_df], axis=1)
                
                # individual graphs measures
                for gr_name in gr_names:
                    gr_df = _graph_measures_to_1D(gr_ms[gr_name], ch_names, filename, \
                                                  ms_conn, bd_name, gr_name)
                    
                    features_row = pd.concat([features_row, gr_df], axis=1)
                
                # conn measures
                for ms_type in ms_types:
                    ms_c = conn_ms[ms_conn][filename][bd_name][ms_type]
                    ft_df = _triangular_to_1D(ms_c, ch_names, filename, \
                                              bd_name, ms_type, ms_conn)
                    features_row = pd.concat([features_row, ft_df], axis=1)
                            
        allFeatures = pd.concat([allFeatures, features_row], axis=0)
    
    return allFeatures.fillna(0)
    

# Make Data Array: Features + Labels
def add_labels_to_data_array(data):
        
    labels = pd.read_excel('Metadata_train.xlsx', index_col='Filename')['Diagnosis']
    
    labels[labels != 'epileptic seizure'] = 0
    labels[labels == 'epileptic seizure'] = 1
   
    data.insert(loc=0, column='y', value=labels)


# double 5-fold nested cross-validation
def dataset_split(data):

    y = data['y'].to_numpy(dtype=float)
    X = data.drop('y', axis=1).to_numpy()
        
    skf = StratifiedKFold(n_splits=5)
        
    datasets = []
    
    for train_index, ts_index in skf.split(X, y):

        X_tr, X_ts = X[train_index], X[ts_index]
        y_tr, y_ts = y[train_index], y[ts_index]
        
        d = {
            'X_tr': X_tr,
            'y_tr': y_tr,
            'X_ts': X_ts,
            'y_ts': y_ts
        }
               
        datasets.append(d)    
                    
    return datasets
