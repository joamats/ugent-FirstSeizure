import numpy as np
import seaborn as sb
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from Pickle import getPickleFile
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

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
    'classifier__C': [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 7, 10, 20, 50, 100],
    'classifier__gamma': [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 7, 10, 20, 50, 100],
    'classifier__kernel': ['rbf', 'sigmoid']#, 'linear']
})

# Feature Selection
dim_red = SelectKBest(score_func=f_classif)

space['dim_red__k'] = [50, 60, 70, 80, 90, 100]

# Pipeline
model_SVC = Pipeline(steps=[('norm_scaler',norm_scaler),
                            ('dim_red', dim_red),
                            ('classifier', svc)])

clf = RandomizedSearchCV(estimator=model_SVC,
                         param_distributions=space,
                         n_iter=100,
                         scoring='roc_auc', 
                         n_jobs=-1,
                         cv=skf,
                         return_train_score=True,
                         random_state=42)

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
    'classifier__C': [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 7, 10, 20, 50, 100],
    'classifier__gamma': [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 7, 10, 20, 50, 100],
    'classifier__kernel': ['rbf', 'sigmoid']#, 'linear']
})

# Dimensionality Reduction
dim_red = PCA(random_state=42)

space['dim_red__n_components'] = [2, 3, 5, 7, 8, 10, 12, 15, 17, 20, 50]

# Pipeline
model_SVC = Pipeline(steps=[('norm_scaler',norm_scaler),
                            ('dim_red', dim_red),
                            ('classifier', svc)])

clf = RandomizedSearchCV(estimator=model_SVC,
                         param_distributions=space,
                         n_iter=100,
                         scoring='roc_auc', 
                         n_jobs=-1,
                         cv=skf,
                         return_train_score=True,
                         random_state=42)

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

# #%% MLP + SelectKBest

# print('\nMLP + SelectKBest\n')

# # Feature Normalization
# norm_scaler = StandardScaler(with_mean=True, with_std=True)
# minMax_scaler = MinMaxScaler()

# # MLP Model
# mlp = MLPClassifier(random_state=42, max_iter = 1000, early_stopping = True)

# # Cross-Validation
# skf = StratifiedKFold(n_splits=5)

# # Parameters for Grid Search
# space = dict({
#     'classifier__hidden_layer_sizes':[(20,), (50,), (100,), (150,), 
#                                      (20,20),(50,50),(100,100), (150,150),
#                                      (20,20,20),(50,50,50), (100,100,100),
#                                      (150,150,150), (500), (1000), (500,500),
#                                      (1000,1000), (500,500,500),(1000,1000,1000)],
#     'classifier__activation': ['relu'],
#     'classifier__solver': ['adam'],
#     'classifier__learning_rate': ["invscaling"],
#     'classifier__alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
# })

# # Feature Selection
# dim_red = SelectKBest(score_func=f_classif)

# space['dim_red__k'] = [50, 60, 70, 80, 90, 100]

# # Pipeline
# model_MLP = Pipeline(steps=[('norm_scaler',norm_scaler),
#                             ('min_max', minMax_scaler),
#                             ('dim_red', dim_red),
#                             ('classifier', mlp)])

# clf = RandomizedSearchCV(estimator=model_MLP,
#                          param_distributions=space,
#                          n_iter=100,
#                          scoring='roc_auc', 
#                          n_jobs=-1,
#                          cv=skf,
#                          return_train_score=True,
#                          random_state=42)

# clfs = []
# scores = []

# for dset in datasets:
#   X_tr = dset['X_tr']
#   y_tr = dset['y_tr']
#   clf.fit(X_tr, y_tr)
#   clfs.append(clf.best_params_)
#   scores.append(clf.best_score_)
#   print('Best Score: ')
#   print(clf.best_score_)
#   print('Best Parameters: ')
#   print(clf.best_params_)

# best_mean_score = np.mean(scores)
# best_std = np.std(scores)
# print('---\nMean Best Score: ', best_mean_score)
# print('\nMean Std Score: ', best_std)

# allVars.append((clfs,scores))

# #%% MLP + PCA

# print('\nMLP + PCA\n')

# # Feature Normalization
# norm_scaler = StandardScaler(with_mean=True, with_std=True)
# minMax_scaler = MinMaxScaler()

# # MLP Model
# mlp = MLPClassifier(random_state=42, max_iter = 1000, early_stopping = True)

# # Cross-Validation
# skf = StratifiedKFold(n_splits=5)

# # Parameters for Grid Search
# space = dict({
#     'classifier__hidden_layer_sizes':[(20,), (50,), (100,), (150,), 
#                                      (20,20),(50,50),(100,100), (150,150),
#                                      (20,20,20),(50,50,50), (100,100,100),
#                                      (150,150,150), (500), (1000), (500,500),
#                                      (1000,1000), (500,500,500),(1000,1000,1000)],
#     'classifier__activation': ['relu'],
#     'classifier__solver': ['adam'],
#     'classifier__learning_rate': ["invscaling"],
#     'classifier__alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
# })

# # Dimensionality Reduction
# dim_red = PCA(random_state=42)

# space['dim_red__n_components'] = [2, 3, 5, 7, 8, 10, 12, 15, 17, 20, 50]

# # Pipeline
# model_MLP = Pipeline(steps=[('norm_scaler',norm_scaler),
#                             ('min_max', minMax_scaler),
#                             ('dim_red', dim_red),
#                             ('classifier', mlp)])

# clf = RandomizedSearchCV(estimator=model_MLP,
#                          param_distributions=space,
#                          n_iter=100,
#                          scoring='roc_auc', 
#                          n_jobs=-1,
#                          cv=skf,
#                          return_train_score=True,
#                          random_state=42)

# clfs = []
# scores = []

# for dset in datasets:
#   X_tr = dset['X_tr']
#   y_tr = dset['y_tr']
#   clf.fit(X_tr, y_tr)
#   clfs.append(clf.best_params_)
#   scores.append(clf.best_score_)
#   print('Best Score: ')
#   print(clf.best_score_)
#   print('Best Parameters: ')
#   print(clf.best_params_)

# best_mean_score = np.mean(scores)
# best_std = np.std(scores)
# print('---\nMean Best Score: ', best_mean_score)
# print('\nMean Std Score: ', best_std)

# allVars.append((clfs,scores))

# #%% RFC + SelectKBest

# print('\nRFC + SelectKBest\n')

# # Feature Normalization
# norm_scaler = StandardScaler(with_mean=True, with_std=True)

# # RFC Model
# rfc = RandomForestClassifier(random_state=42)

# # Cross-Validation
# skf = StratifiedKFold(n_splits=5)

# # Parameters for Grid Search
# space = dict({
#     'classifier__bootstrap': [True, False],
#     'classifier__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
#     'classifier__max_features': ['auto', 'sqrt'],
#     'classifier__min_samples_leaf': [1, 2, 4],
#     'classifier__min_samples_split': [2, 5, 10],
#     'classifier__n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
#     'classifier__criterion': ('gini', 'entropy')
# })

# # Feature Selection
# dim_red = SelectKBest(score_func=f_classif)

# space['dim_red__k'] = [50, 60, 70, 80, 90, 100]

# # Pipeline
# model_RFC = Pipeline(steps=[('norm_scaler',norm_scaler),
#                             ('dim_red', dim_red),
#                             ('classifier', rfc)])

# clf = RandomizedSearchCV(estimator=model_RFC,
#                          param_distributions=space,
#                          n_iter=100,
#                          scoring='roc_auc', 
#                          n_jobs=-1,
#                          cv=skf,
#                          return_train_score=True,
#                          random_state=42)

# clfs = []
# scores = []

# for dset in datasets:
#   X_tr = dset['X_tr']
#   y_tr = dset['y_tr']
#   clf.fit(X_tr, y_tr)
#   clfs.append(clf.best_params_)
#   scores.append(clf.best_score_)
#   print('Best Score: ')
#   print(clf.best_score_)
#   print('Best Parameters: ')
#   print(clf.best_params_)

# best_mean_score = np.mean(scores)
# best_std = np.std(scores)
# print('---\nMean Best Score: ', best_mean_score)
# print('\nMean Std Score: ', best_std)

# allVars.append((clfs,scores))

# #%% RFC + PCA

# print('\nRFC + PCA\n')

# # Feature Normalization
# norm_scaler = StandardScaler(with_mean=True, with_std=True)

# # RFC Model
# rfc = RandomForestClassifier(random_state=42)

# # Cross-Validation
# skf = StratifiedKFold(n_splits=5)

# # Parameters for Grid Search
# space = dict({
#     'classifier__bootstrap': [True, False],
#     'classifier__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
#     'classifier__max_features': ['auto', 'sqrt'],
#     'classifier__min_samples_leaf': [1, 2, 4],
#     'classifier__min_samples_split': [2, 5, 10],
#     'classifier__n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
#     'classifier__criterion': ('gini', 'entropy')
# })

# # Dimensionality Reduction
# dim_red = PCA(random_state=42)

# space['dim_red__n_components'] = [2, 3, 5, 7, 8, 10, 12, 15, 17, 20, 50]

# # Pipeline
# model_RFC = Pipeline(steps=[('norm_scaler',norm_scaler),
#                             ('dim_red', dim_red),
#                             ('classifier', rfc)])

# clf = RandomizedSearchCV(estimator=model_RFC,
#                          param_distributions=space,
#                          n_iter=100,
#                          scoring='roc_auc', 
#                          n_jobs=-1,
#                          cv=skf,
#                          return_train_score=True,
#                          random_state=42)

# clfs = []
# scores = []

# for dset in datasets:
#   X_tr = dset['X_tr']
#   y_tr = dset['y_tr']
#   clf.fit(X_tr, y_tr)
#   clfs.append(clf.best_params_)
#   scores.append(clf.best_score_)
#   print('Best Score: ')
#   print(clf.best_score_)
#   print('Best Parameters: ')
#   print(clf.best_params_)

# best_mean_score = np.mean(scores)
# best_std = np.std(scores)
# print('---\nMean Best Score: ', best_mean_score)
# print('\nMean Std Score: ', best_std)

# allVars.append((clfs,scores))
