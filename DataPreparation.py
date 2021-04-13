from Pickle import getPickleFile, createPickleFile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#%% Auxiliary functions

# retrieves all saved features (conn + graphs)
def get_saved_features(bdp=False, rawConn=False, conn=False, graphs=False, asy=False):
    
    features = []
    
    if bdp:
        features.append(getPickleFile('../2_Features_Data/128Hz/' + 'bdp'))
    
    if rawConn:
        IMCOH = getPickleFile('../2_Features_Data/128Hz/' + 'imcoh')
        PLV = getPickleFile('../2_Features_Data/128Hz/' + 'plv')
        MI = getPickleFile('../2_Features_Data/128Hz/' + 'mi')
        PDC = getPickleFile('../2_Features_Data/128Hz/' + 'pdc')
        features.append({'imcoh': IMCOH, 'plv': PLV, 'mi': MI, 'pdc': PDC})
    
    if conn:
        features.append(getPickleFile('../2_Features_Data/128Hz/' + 'connectivityMeasures'))
    
    if graphs:
        features.append(getPickleFile('../2_Features_Data/128Hz/' + 'graphMeasures'))
        
    if asy:    
        features.append(getPickleFile('../2_Features_Data/128Hz/' + 'asymmetryMeasures'))

    if len(features)==1:
        return features[0]
    else:
        return features

      
#%% 

# Produce Features Array for ML
def make_features_array(filenames, bdp_ms, conn_ms, gr_ms, asy_ms):
            
    allFeatures = pd.DataFrame()
    
    for filename in filenames:
        
        features_row = pd.DataFrame()
        # concatenate bandpowers
        ft_df = bdp_ms[filename].T
        features_row = pd.concat([features_row, ft_df], axis=1)
        # concatenate connectivity measures
        ft_df = conn_ms[filename].T
        features_row = pd.concat([features_row, ft_df], axis=1)
        # concatenate graph measures
        ft_df = gr_ms[filename].T
        features_row = pd.concat([features_row, ft_df], axis=1)
        # concatenate asymmetry measures
        ft_df = asy_ms[filename].T
        features_row = pd.concat([features_row, ft_df], axis=1)
        # join this subject's row to all subjects
        allFeatures = pd.concat([allFeatures, features_row], axis=0)
    
    return allFeatures.fillna(0)

# Get filenames and labels from metadata
def get_filenames_labels(mode='Diagnosis'):
    
    if mode == 'Diagnosis':
        filenames = pd.read_excel('Metadata_train.xlsx')['Filename']
        labels = pd.read_excel('Metadata_train.xlsx', index_col='Filename')['Diagnosis']
        
    elif mode == 'Epilepsy types':
        meta_labels = pd.read_excel('Metadata_train.xlsx', index_col='Filename')['Epilepsy type']
        labels = meta_labels[~ meta_labels.isnull()]
        labels = labels[labels != 'undetermined']
        filenames = labels.index
    
    elif mode == 'Gender':
        filenames = pd.read_excel('Metadata_train.xlsx')['Filename']
        labels = pd.read_excel('Metadata_train.xlsx', index_col='Filename')['Gender']
        
    return labels, filenames

# Make Data Array: Features + Labels
def add_labels_to_data_array(data, labels, mode='Diagnosis'):
        
    if mode == 'Diagnosis':
        labels[labels != 'epileptic seizure'] = 0
        labels[labels == 'epileptic seizure'] = 1

    elif mode == 'Epilepsy types':
        labels[labels == 'cryptogenic'] = 0
        labels[labels == 'focal cryptogenic'] = 0
        labels[labels == 'focal symptomatic'] = 1
        labels[labels == 'generalized idiopathic'] = 2
        
    elif mode == 'Gender':
        labels[labels == 'female'] = 0
        labels[labels == 'male'] = 1
    
    
    data.insert(loc=0, column='y', value=labels)


# double 5-fold nested cross-validation
def dataset_split(data):

    y = data['y'].to_numpy(dtype=float)
    X = data.drop('y', axis=1).to_numpy()
        
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.20, shuffle=True, random_state=42)
        
    return {'X_tr': X_tr, 'X_ts': X_ts, 'y_tr': y_tr, 'y_ts': y_ts}
