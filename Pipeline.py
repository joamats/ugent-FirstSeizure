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
from FeatureExtraction import extract_features
from GraphMeasures import compute_graph_measures

from DataPreparation import get_saved_features,  make_features_array, \
                            add_labels_to_data_array, dataset_split

#%% Save Features - just change PDC's order to 12
filenames = pd.read_excel('Metadata_train.xlsx')['Filename']

# IMCOH = {}
# PLV = {}
# MI = {}
PDC = {}

# over all subjects
for i, filename in enumerate(filenames):
    saved_epochs = getPickleFile('../PreProcessed_Data/128Hz/' + filename)
    bd_names, s_epochs = epochs_selection_bandpower(saved_epochs)
    # IMCOH[filename], PLV[filename], MI[filename],\
    PDC[filename] = extract_features(bd_names, s_epochs)
    
    # save features in pickle
    # createPickleFile(IMCOH, '../Features/128Hz/' + 'imcoh')
    # createPickleFile(PLV, '../Features/128Hz/' + 'plv')
    # createPickleFile(MI, '../Features/128Hz/' + 'mi')
    createPickleFile(PDC, '../Features/128Hz/' + 'pdc')         

#%% Graph Measures

fts = get_saved_features(withGraphs=False)
graph_ms = compute_graph_measures(fts)
createPickleFile(graph_ms, '../Features/128Hz/' + 'graphMeasures')

#%% Save Data
conn_ms, graph_ms = get_saved_features(withGraphs=True)

data = make_features_array(conn_ms, graph_ms, std = True)
fts_names = data.columns

createPickleFile(data, '../Features/128Hz/' + 'allFeatures')
createPickleFile(fts_names, '../ML_Data/128Hz/' + 'featuresNames')

add_labels_to_data_array(data)
datasets = dataset_split(data)

createPickleFile(datasets, '../ML_Data/128Hz/' + 'datasets')

#%%
datasets = getPickleFile('../ML_Data/128Hz/datasets')
print('datasets loaded')

allVars = []

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

clfs = []
scores = []

for dset in datasets:
  X_tr = dset['X_tr']
  y_tr = dset['y_tr']
  clf.fit(X_tr, y_tr)
  clfs.append(clf.best_params_)
  scores.append(clf.best_score_)
  print('Best Score: ')
  print(clf.best_score_)
  print('Best Parameters: ')
  print(clf.best_params_)

best_mean_score = np.mean(scores)
best_std = np.std(scores)
print('---\nMean Best Score: ', best_mean_score)
print('\nMean Std Score: ', best_std)

allVars.append((clfs,scores))


#%% SVM + PCA

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

for dset in datasets:
  X_tr = dset['X_tr']
  y_tr = dset['y_tr']
  clf.fit(X_tr, y_tr)
  clfs.append(clf.best_params_)
  scores.append(clf.best_score_)
  print('Best Score: ')
  print(clf.best_score_)
  print('Best Parameters: ')
  print(clf.best_params_)

best_mean_score = np.mean(scores)
best_std = np.std(scores)
print('---\nMean Best Score: ', best_mean_score)
print('\nMean Std Score: ', best_std)

allVars.append((clfs,scores))

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

clfs = []
scores = []

for dset in datasets:
  X_tr = dset['X_tr']
  y_tr = dset['y_tr']
  clf.fit(X_tr, y_tr)
  clfs.append(clf.best_params_)
  scores.append(clf.best_score_)
  print('Best Score: ')
  print(clf.best_score_)
  print('Best Parameters: ')
  print(clf.best_params_)

best_mean_score = np.mean(scores)
best_std = np.std(scores)
print('---\nMean Best Score: ', best_mean_score)
print('\nMean Std Score: ', best_std)

allVars.append((clfs,scores))

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

clfs = []
scores = []

for dset in datasets:
  X_tr = dset['X_tr']
  y_tr = dset['y_tr']
  clf.fit(X_tr, y_tr)
  clfs.append(clf.best_params_)
  scores.append(clf.best_score_)
  print('Best Score: ')
  print(clf.best_score_)
  print('Best Parameters: ')
  print(clf.best_params_)

best_mean_score = np.mean(scores)
best_std = np.std(scores)
print('---\nMean Best Score: ', best_mean_score)
print('\nMean Std Score: ', best_std)

allVars.append((clfs,scores))

# RFC + SelectKBest

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

clfs = []
scores = []

for dset in datasets:
  X_tr = dset['X_tr']
  y_tr = dset['y_tr']
  clf.fit(X_tr, y_tr)
  clfs.append(clf.best_params_)
  scores.append(clf.best_score_)
  print('Best Score: ')
  print(clf.best_score_)
  print('Best Parameters: ')
  print(clf.best_params_)

best_mean_score = np.mean(scores)
best_std = np.std(scores)
print('---\nMean Best Score: ', best_mean_score)
print('\nMean Std Score: ', best_std)

allVars.append((clfs,scores))

# RFC + PCA

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

clfs = []
scores = []

for dset in datasets:
  X_tr = dset['X_tr']
  y_tr = dset['y_tr']
  clf.fit(X_tr, y_tr)
  clfs.append(clf.best_params_)
  scores.append(clf.best_score_)
  print('Best Score: ')
  print(clf.best_score_)
  print('Best Parameters: ')
  print(clf.best_params_)

best_mean_score = np.mean(scores) 
best_std = np.std(scores)
print('---\nMean Best Score: ', best_mean_score)
print('\nMean Std Score: ', best_std)

allVars.append((clfs,scores))
