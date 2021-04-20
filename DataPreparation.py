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
        features_row = features_row[features_row.columns.drop(list(features_row.filter(regex='mi')))]
        allFeatures = pd.concat([allFeatures, features_row], axis=0)
    
    return allFeatures.fillna(0)

# Get filenames and labels from metadata
def get_filenames_labels(mode='Diagnosis'):
    
    if mode == 'Diagnosis' or mode =='DiagnosisWithAgeGender':
        meta_labels = pd.read_excel('Metadata_train.xlsx', index_col='Filename')[['Diagnosis','Sleep state']]
        labels = meta_labels[~ meta_labels.isnull()]
        labels = labels[labels['Sleep state'] == 'wake']
        labels = labels[labels != 'undetermined']
        labels = labels[~ labels['Diagnosis'].isnull()]
        labels = labels['Diagnosis']
        filenames = labels.index
        
    elif mode == 'Epilepsy types':
        meta_labels = pd.read_excel('Metadata_train.xlsx', index_col='Filename')['Epilepsy type']
        labels = meta_labels[~ meta_labels.isnull()]
        labels = labels[labels != 'undetermined']
        filenames = labels.index
    
    elif mode == 'Gender':
        filenames = pd.read_excel('Metadata_train.xlsx')['Filename']
        labels = pd.read_excel('Metadata_train.xlsx', index_col='Filename')['Gender']
        
    elif mode == 'Age':
        filenames = pd.read_excel('Metadata_train.xlsx')['Filename']
        labels = pd.read_excel('Metadata_train.xlsx', index_col='Filename')['Age']
        
    elif mode == 'Sleep':
        meta_labels = pd.read_excel('Metadata_train.xlsx', index_col='Filename')['Sleep state']
        labels = meta_labels[~ meta_labels.isnull()]
        filenames = labels.index
    
    elif mode == 'Diagnosis-Sleep':
        meta_labels = pd.read_excel('Metadata_train.xlsx', index_col='Filename')[['Diagnosis','Sleep state']]
        labels = meta_labels[~ meta_labels.isnull()]
        labels = labels[labels['Diagnosis'] == 'epileptic seizure']['Sleep state']
        labels = labels[~ labels.isnull()]
        filenames = labels.index
        
    elif mode == 'CardiovascularVSEpileptic':
        meta_labels = pd.read_excel('Metadata_train.xlsx', index_col='Filename')[['Diagnosis','Sleep state']]
        labels = meta_labels[~ meta_labels.isnull()]
        labels = labels[labels['Sleep state'] == 'wake']
        labels = labels[labels != 'undetermined']
        labels = labels[labels != 'provoked seizure']
        labels = labels[labels != 'vagal syncope']
        labels = labels[labels != 'other']
        labels = labels[labels != 'psychogenic']
        labels = labels[~ labels['Sleep state'].isnull()]
        labels = labels[~ labels['Diagnosis'].isnull()]
        labels = labels['Diagnosis']
        filenames = labels.index

    elif mode == 'ProvokedVSEpileptic':
        meta_labels = pd.read_excel('Metadata_train.xlsx', index_col='Filename')[['Diagnosis','Sleep state']]
        labels = meta_labels[~ meta_labels.isnull()]
        labels = labels[labels['Sleep state'] == 'wake']
        labels = labels[labels != 'undetermined']
        labels = labels[labels != 'cardiovascular']
        labels = labels[labels != 'vagal syncope']
        labels = labels[labels != 'other']
        labels = labels[labels != 'psychogenic']
        labels = labels[~ labels['Sleep state'].isnull()]
        labels = labels[~ labels['Diagnosis'].isnull()]
        labels = labels['Diagnosis']

        filenames = labels.index
        
    elif mode == 'PsychogenicVSEpileptic':
        meta_labels = pd.read_excel('Metadata_train.xlsx', index_col='Filename')[['Diagnosis','Sleep state']]
        labels = meta_labels[~ meta_labels.isnull()]
        labels = labels[labels['Sleep state'] == 'wake']
        labels = labels[labels != 'undetermined']
        labels = labels[labels != 'cardiovascular']
        labels = labels[labels != 'vagal syncope']
        labels = labels[labels != 'other']
        labels = labels[labels != 'provoked seizure']
        labels = labels[~ labels['Sleep state'].isnull()]
        labels = labels[~ labels['Diagnosis'].isnull()]
        labels = labels['Diagnosis']
        
        filenames = labels.index
        
    elif mode == 'VagalSyncopeVSEpileptic':
        meta_labels = pd.read_excel('Metadata_train.xlsx', index_col='Filename')[['Diagnosis','Sleep state']]
        labels = meta_labels[~ meta_labels.isnull()]
        labels = labels[labels['Sleep state'] == 'wake']
        labels = labels[labels != 'undetermined']
        labels = labels[labels != 'cardiovascular']
        labels = labels[labels != 'psychogenic']
        labels = labels[labels != 'other']
        labels = labels[labels != 'provoked seizure']
        labels = labels[~ labels['Sleep state'].isnull()]
        labels = labels[~ labels['Diagnosis'].isnull()]
        labels = labels['Diagnosis']
        
        filenames = labels.index
        
    elif mode == 'AntecedentFamilyEpileptic':
        meta_labels = pd.read_excel('Metadata_train.xlsx', index_col='Filename')[['Diagnosis','Sleep state', 'Antecedent family']]
        labels = meta_labels[~ meta_labels.isnull()]
        labels = labels[labels['Sleep state'] == 'wake']
        labels = labels[labels['Antecedent family'] == 'epilepsy']
        labels = labels[labels != 'undetermined']
        labels = labels[~ labels['Diagnosis'].isnull()]
        labels = labels['Diagnosis']
        
        filenames = labels.index
        
    elif mode == 'AntecedentFamilyNonEpileptic':
        meta_labels = pd.read_excel('Metadata_train.xlsx', index_col='Filename')[['Diagnosis','Sleep state', 'Antecedent family']]
        labels = meta_labels[~ meta_labels.isnull()]
        labels = labels[labels['Sleep state'] == 'wake']
        labels = labels[labels['Antecedent family'] == 'none']
        labels = labels[labels != 'undetermined']
        labels = labels[~ labels['Diagnosis'].isnull()]
        labels = labels['Diagnosis']
        
        filenames = labels.index
        
    elif mode == 'AntecedentFamilyOther':
        meta_labels = pd.read_excel('Metadata_train.xlsx', index_col='Filename')[['Diagnosis','Sleep state', 'Antecedent family']]
        labels = meta_labels[~ meta_labels.isnull()]
        labels = labels[labels['Sleep state'] == 'wake']
        labels = labels[labels['Antecedent family'] == 'other']
        labels = labels[labels != 'undetermined']
        labels = labels[~ labels['Diagnosis'].isnull()]
        labels = labels['Diagnosis']
        
        filenames = labels.index
        
    return labels, filenames

# Make Data Array: Features + Labels
def add_labels_to_data_array(data, labels, mode='Diagnosis'):
        
    flt_labels = labels.copy()
    
    if mode == 'Diagnosis' or mode =='DiagnosisWithAgeGender' or mode == 'AntecedentFamilyEpileptic' or mode == 'AntecedentFamilyNonEpileptic' or mode == 'AntecedentFamilyOther':
        flt_labels[labels != 'epileptic seizure'] = 0
        flt_labels[labels == 'epileptic seizure'] = 1
        
        labels_names = ['non-epileptic', 'epileptic seizure']

    elif mode == 'Epilepsy types':
        flt_labels[labels == 'cryptogenic'] = 0
        flt_labels[labels == 'focal cryptogenic'] = 0
        flt_labels[labels == 'focal symptomatic'] = 1
        flt_labels[labels == 'generalized idiopathic'] = 2
        
        labels_names = ['cryptogenic', 'focal symptomatic', 'generalized idiopathic']
        
    elif mode == 'Gender':
        flt_labels[labels == 'female'] = 0
        flt_labels[labels == 'male'] = 1
        
        labels_names = ['female', 'male']
        
    elif mode == 'Age':
        flt_labels[labels < 50] = 0
        flt_labels[labels >= 50 ] = 1
        
        labels_names = ['young', 'old']
        
    elif mode == 'Sleep':
        flt_labels[labels == 'sleep'] = 0
        flt_labels[labels == 'wake'] = 1
        
        labels_names = ['sleep', 'wake']
        
    elif mode == 'Diagnosis-Sleep':
        flt_labels[labels == 'sleep'] = 0
        flt_labels[labels == 'wake'] = 1
        
        labels_names = ['sleep', 'wake']
        
    elif mode == 'CardiovascularVSEpileptic':
        flt_labels[labels == 'cardiovascular'] = 0
        flt_labels[labels == 'epileptic seizure'] = 1
        
        labels_names = ['cardiovascular', 'epileptic seizure']

    elif mode == 'ProvokedVSEpileptic':
        flt_labels[labels == 'provoked seizure'] = 0
        flt_labels[labels == 'epileptic seizure'] = 1
        
        labels_names = ['provoked seizure', 'epileptic seizure']
        
    elif mode == 'PsychogenicVSEpileptic':
        flt_labels[labels == 'psychogenic'] = 0
        flt_labels[labels == 'epileptic seizure'] = 1
        
        labels_names = ['psychogenic', 'epileptic seizure']
        
    elif mode == 'VagalSyncopeVSEpileptic':
        flt_labels[labels == 'vagal syncope'] = 0
        flt_labels[labels == 'epileptic seizure'] = 1
        
        labels_names = ['vagal syncope', 'epileptic seizure']


    data.insert(loc=0, column='y', value=flt_labels)
    
    return labels_names


# 5-fold cross-validation
def dataset_split(data):

    y = data['y'].to_numpy(dtype=float)
    X = data.drop('y', axis=1).to_numpy()
        
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.20, shuffle=True, random_state=42)
        
    return {'X_tr': X_tr, 'X_ts': X_ts, 'y_tr': y_tr, 'y_ts': y_ts}
