import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from Pickle import getPickleFile, createPickleFile
from PreProcessing import epochs_selection_bandpower
from FeatureExtraction import extract_bandpowers, extract_features, compute_connectivity_measures
from GraphMeasures import compute_graph_subgroup_measures
from Asymmetry import compute_asymmetry_measures

from DataPreparation import get_saved_features,  make_features_array, \
                            add_labels_to_data_array, dataset_split

'''
    Subgroupings of connectivity matrix and graph measures
'''

#%% Bandpower and Connectivity Features 
filenames = pd.read_excel('Metadata_train.xlsx')['Filename']

BDP = {}
IMCOH = {}
PLV = {}
MI = {}
PDC = {}

# over all subjects
for i, filename in enumerate(filenames[[0]]):
    saved_epochs = getPickleFile('../1_PreProcessed_Data/128Hz/' + filename)
        
    BDP[filename] = extract_bandpowers(saved_epochs, filename)
    
    bd_names, s_epochs = epochs_selection_bandpower(saved_epochs)
    
    IMCOH[filename], PLV[filename], MI[filename],\
    PDC[filename] = extract_features(bd_names, s_epochs)
    
    # save features in pickle
    createPickleFile(BDP, '../2_Features_Data/128Hz/' + 'bdp')
    createPickleFile(IMCOH, '../2_Features_Data/128Hz/' + 'imcoh')
    createPickleFile(PLV, '../2_Features_Data/128Hz/' + 'plv')
    createPickleFile(MI, '../2_Features_Data/128Hz/' + 'mi')
    createPickleFile(PDC, '../2_Features_Data/128Hz/' + 'pdc')         

#%% Subgroups Connectivity Features
fts = get_saved_features(bdp=False, rawConn=True, conn=False, graphs=False, asy=False)
conn_ms = compute_connectivity_measures(fts)
createPickleFile(conn_ms, '../2_Features_Data/128Hz/' + 'connectivityMeasures')

#%% Subgroups Graph Measures
fts = get_saved_features(bdp=False, rawConn=True, conn=False, graphs=False, asy=False)
graph_ms = compute_graph_subgroup_measures(fts)
createPickleFile(graph_ms, '../2_Features_Data/128Hz/' + 'graphMeasures')

#%% Subgroups Graph Asymmetry Ratios
fts = get_saved_features(bdp=False, rawConn=False, conn=False, graphs=True, asy=False)
asymmetry_ms = compute_asymmetry_measures(fts)
createPickleFile(asymmetry_ms, '../2_Features_Data/128Hz/' + 'asymmetryMeasures')

#%% Generate All Features Matrix
bdp_ms, conn_ms, gr_ms = get_saved_features(bdp=True, rawConn=False, conn=True, graphs=True, asy=False)
data = make_features_array(bdp_ms, conn_ms, gr_ms)
fts_names = data.columns

createPickleFile(data, '../2_Features_Data/128Hz/' + 'allFeatures')
createPickleFile(fts_names, '../3_ML_Data/128Hz/' + 'featuresNames')

add_labels_to_data_array(data)
dataset = dataset_split(data)

createPickleFile(dataset, '../3_ML_Data/128Hz/' + 'dataset')

#%% Get Dataset
dataset = getPickleFile('../3_ML_Data/128Hz/dataset')

#%% SVM + SelectKBest

print('\nSVM + SelectKBest\n')

# Feature Normalization
norm_scaler = StandardScaler(with_mean=True, with_std=True)

# SVC Model
svc = SVC(random_state=42)

# Cross-Validation
skf = StratifiedKFold(n_splits=5)

# Parameters for Grid Search
space = dict({
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__gamma': [0.01, 0.1, 1, 10, 100],
    'classifier__kernel': ['rbf', 'linear', 'sigmoid']
})

# Feature Selection
dim_red = SelectKBest(score_func=f_classif)

space['dim_red__k'] = [20, 50, 70]

# Pipeline
model_SVC = Pipeline(steps=[('norm_scaler',norm_scaler),
                            ('dim_red', dim_red),
                            ('classifier', svc)])

clf = GridSearchCV( estimator=model_SVC,
                    param_grid=space,
                    scoring='roc_auc', 
                    n_jobs=-1,
                    cv=skf,
                    return_train_score=True )

X_tr = dataset['X_tr']
y_tr = dataset['y_tr']
clf.fit(X_tr, y_tr)
print('Best Score: ')
print(clf.best_score_)
print('Best Parameters: ')
print(clf.best_params_)



#% SVM + PCA

print('\nSVM + PCA\n')

# Feature Normalization
norm_scaler = StandardScaler(with_mean=True, with_std=True)

# SVC Model
svc = SVC(random_state=42)

# Cross-Validation
skf = StratifiedKFold(n_splits=5)

# Parameters for Grid Search
space = dict({
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__gamma': [0.01, 0.1, 1, 10, 100],
    'classifier__kernel': ['rbf', 'linear', 'sigmoid']
})

# Dimensionality Reduction
dim_red = PCA(random_state=42)

space['dim_red__n_components'] = [10, 15, 20]

# Pipeline
model_SVC = Pipeline(steps=[('norm_scaler',norm_scaler),
                            ('dim_red', dim_red),
                            ('classifier', svc)])

clf = GridSearchCV( estimator=model_SVC,
                    param_grid=space,
                    scoring='roc_auc', 
                    n_jobs=-1,
                    cv=skf,
                    return_train_score=True )

clfs = []
scores = []

X_tr = dataset['X_tr']
y_tr = dataset['y_tr']
clf.fit(X_tr, y_tr)
print('Best Score: ')
print(clf.best_score_)
print('Best Parameters: ')
print(clf.best_params_)

#%% MLP + SelectKBest

print('\nMLP + SelectKBest\n')

# Feature Normalization
norm_scaler = StandardScaler(with_mean=True, with_std=True)
minMax_scaler = MinMaxScaler()

# MLP Model
mlp = MLPClassifier(random_state=42, max_iter = 1000, early_stopping = True)

# Cross-Validation
skf = StratifiedKFold(n_splits=5)

# Parameters for Grid Search
space = dict({
    'classifier__hidden_layer_sizes':[(100), (150), (500), (1000),
                                      (50,50), (100,100), (150,150),(500,500),
                                      (50,50,50), (100,100,100),(150,150,150)],
    'classifier__activation': ['relu'],
    'classifier__solver': ['adam'],
    'classifier__learning_rate': ['adaptive'],
    'classifier__alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
    'classifier__early_stopping': [True, False]
})

# Feature Selection
dim_red = SelectKBest(score_func=f_classif)

space['dim_red__k'] = [20, 50, 70]

# Pipeline
model_MLP = Pipeline(steps=[('norm_scaler',norm_scaler),
                            ('min_max', minMax_scaler),
                            ('dim_red', dim_red),
                            ('classifier', mlp)])

clf = GridSearchCV( estimator=model_MLP,
                    param_grid=space,
                    scoring='roc_auc', 
                    n_jobs=-1,
                    cv=skf,
                    return_train_score=True )

X_tr = dataset['X_tr']
y_tr = dataset['y_tr']
clf.fit(X_tr, y_tr)
print('Best Score: ')
print(clf.best_score_)
print('Best Parameters: ')
print(clf.best_params_)

#%% MLP + PCA

print('\nMLP + PCA\n')

# Feature Normalization
norm_scaler = StandardScaler(with_mean=True, with_std=True)
minMax_scaler = MinMaxScaler()

# MLP Model
mlp = MLPClassifier(random_state=42, max_iter = 1000)

# Cross-Validation
skf = StratifiedKFold(n_splits=5)

# Parameters for Grid Search
space = dict({
    'classifier__hidden_layer_sizes':[(100), (150), (500), (1000),
                                      (50,50), (100,100), (150,150),(500,500),
                                      (50,50,50), (100,100,100),(150,150,150)],
    'classifier__activation': ['relu'],
    'classifier__solver': ['adam'],
    'classifier__learning_rate': ['adaptive'],
    'classifier__alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
    'classifier__early_stopping': [True, False]
})

# Dimensionality Reduction
dim_red = PCA(random_state=42)

space['dim_red__n_components'] = [10, 15, 20]


# Pipeline
model_MLP = Pipeline(steps=[('norm_scaler',norm_scaler),
                            ('min_max', minMax_scaler),
                            ('dim_red', dim_red),
                            ('classifier', mlp)])

clf = GridSearchCV( estimator=model_MLP,
                    param_grid=space,
                    scoring='roc_auc', 
                    n_jobs=-1,
                    cv=skf,
                    return_train_score=True )

X_tr = dataset['X_tr']
y_tr = dataset['y_tr']
clf.fit(X_tr, y_tr)
print('Best Score: ')
print(clf.best_score_)
print('Best Parameters: ')
print(clf.best_params_)

#%% RFC + SelectKBest

print('\nRFC + SelectKBest\n')

# Feature Normalization
norm_scaler = StandardScaler(with_mean=True, with_std=True)

# RFC Model
rfc = RandomForestClassifier(random_state=42)

# Cross-Validation
skf = StratifiedKFold(n_splits=5)

# Parameters for Grid Search
space = dict({
    'classifier__bootstrap': [True],
    'classifier__max_depth': [50, 70, 90, None],
    'classifier__max_features': ['auto'],
    'classifier__min_samples_leaf': [1, 5],
    'classifier__min_samples_split': [2, 5],
    'classifier__n_estimators': [500, 1000, 1500],
    'classifier__criterion': ['gini']
})

# Feature Selection
dim_red = SelectKBest(score_func=f_classif)

space['dim_red__k'] = [20, 50, 70]

# Pipeline
model_RFC = Pipeline(steps=[('norm_scaler',norm_scaler),
                            ('dim_red', dim_red),
                            ('classifier', rfc)])

clf = GridSearchCV( estimator=model_RFC,
                    param_grid=space,
                    scoring='roc_auc', 
                    n_jobs=-1,
                    cv=skf,
                    return_train_score=True )

X_tr = dataset['X_tr']
y_tr = dataset['y_tr']
clf.fit(X_tr, y_tr)
print('Best Score: ')
print(clf.best_score_)
print('Best Parameters: ')
print(clf.best_params_)

#%% RFC + PCA

print('\nRFC + PCA\n')

# Feature Normalization
norm_scaler = StandardScaler(with_mean=True, with_std=True)

# RFC Model
rfc = RandomForestClassifier(random_state=42)

# Cross-Validation
skf = StratifiedKFold(n_splits=5)

# Parameters for Grid Search
space = dict({
    'classifier__bootstrap': [True],
    'classifier__max_depth': [50, 70, 90, None],
    'classifier__max_features': ['auto'],
    'classifier__min_samples_leaf': [1, 5],
    'classifier__min_samples_split': [2, 5],
    'classifier__n_estimators': [500, 1000, 1500],
    'classifier__criterion': ['gini']
})

# Dimensionality Reduction
dim_red = PCA(random_state=42)

space['dim_red__n_components'] = [10, 15, 20]

# Pipeline
model_RFC = Pipeline(steps=[('norm_scaler',norm_scaler),
                            ('dim_red', dim_red),
                            ('classifier', rfc)])

clf = GridSearchCV( estimator=model_RFC,
                    param_grid=space,
                    scoring='roc_auc', 
                    n_jobs=-1,
                    cv=skf,
                    return_train_score=True )

X_tr = dataset['X_tr']
y_tr = dataset['y_tr']
clf.fit(X_tr, y_tr)
print('Best Score: ')
print(clf.best_score_)
print('Best Parameters: ')
print(clf.best_params_)
