from Pickle import getPickleFile
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
# import seaborn as sb
# import pandas as pd

datasets = getPickleFile('../ML_Data/' + 'datasets')

clf_knn_list=[]

from sklearn.model_selection import GridSearchCV


datasets = getPickleFile('../ML_Data/' + 'datasets')

clf_SVC_s = []
clf_RFC_s = []
clf_MLP_s = []


for dset in datasets:

    X_tr = dset['X_tr']
    y_tr = dset['y_tr']
   
    #Feature Normalization
    norm_scaler = StandardScaler(with_mean=True, with_std=True)
    minMax_scaler = MinMaxScaler()


    # #Feature Selection
    # _k_features = 200
    
    selector = SelectKBest(score_func=f_classif)
    # selector = PCA(random_state=42)
    
    # # models
    # svc = SVC(random_state=42, verbose = True)
    # model_SVC = Pipeline(steps=[('norm_scaler',norm_scaler),
    #                             ('minMax_scaler', minMax_scaler),
    #                             ('selector', selector),
    #                             ('classifier', svc)])
    # rfc = RandomForestClassifier(random_state=42, verbose = True)
    # model_RFC = Pipeline(steps=[('norm_scaler',norm_scaler),
    #                             ('minMax_scaler', minMax_scaler),
    #                             ('selector', selector),
    #                             ('classifier', rfc)])
    # mlp = MLPClassifier(random_state=42, max_iter = 1000, early_stopping = True, verbose = True)
    # model_MLP = Pipeline(steps=[('norm_scaler',norm_scaler),
    #                             ('minMax_scaler', minMax_scaler),
    #                             ('selector', selector),
    #                             ('classifier', mlp)])
    knn = KNeighborsClassifier()
    model_knn = Pipeline(steps=[('norm_scaler',norm_scaler),
                                
    selector = SelectKBest(score_func=f_classif, k=500)
    
    # models
    svc = SVC(random_state=42, verbose = True)
    model_SVC = Pipeline(steps=[('norm_scaler',norm_scaler),
                                ('minMax_scaler', minMax_scaler),
                                ('selector', selector),
                                ('classifier', svc)])
    
    rfc = RandomForestClassifier(random_state=42, verbose = True)
    model_RFC = Pipeline(steps=[('norm_scaler',norm_scaler),
                                ('minMax_scaler', minMax_scaler),
                                ('selector', selector),
                                ('classifier', rfc)])
    
    mlp = MLPClassifier(random_state=42, max_iter = 1000, early_stopping = True, verbose = 10,
                        activation='relu', solver='adam', hidden_layer_sizes=(200, 200),
                        alpha=1e-5, learning_rate='adaptive')
    
    model_MLP = Pipeline(steps=[('norm_scaler', norm_scaler),

                                ('minMax_scaler', minMax_scaler),
                                ('selector', selector),
                                ('classifier', knn)])
    
    # Cross-Validation
    skf = StratifiedKFold(n_splits=5)

    #GridSearchCV

    # parameters={'selector__k': [10, 20, 50, 100, 150, 200, 300, 500, 700, 1000, 1500],
    #             'classifier__kernel': ('rbf', 'poly', 'sigmoid'),
    #             'classifier__degree': [2,3,4,5],
    #             'classifier__C': [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 7, 10, 20, 50, 100],
    #             'classifier__gamma': [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 7, 10, 20, 50, 100]}
    
    # clf_SVC = RandomizedSearchCV(model_SVC, parameters, n_iter = 200, scoring='accuracy', n_jobs=-1, cv=skf)
    # clf_SVC.fit(X_tr,y_tr)
    
    # parameters={'selector__k': [10, 20, 50, 100, 150, 200, 300, 500, 700, 1000, 1500],
    #             'classifier__bootstrap': [True, False],
    #             'classifier__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    #             'classifier__max_features': ['auto', 'sqrt'],
    #             'classifier__min_samples_leaf': [1, 2, 4],
    #             'classifier__min_samples_split': [2, 5, 10],
    #             'classifier__n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
    #             'classifier__criterion': ('gini', 'entropy')}
    
    # clf_RFC = RandomizedSearchCV(model_RFC, parameters, n_iter = 200, scoring='accuracy', n_jobs=-1, cv=skf)
    # clf_RFC.fit(X_tr,y_tr)
        
    # parameters={'selector__k': [50, 70, 90, 100, 300, 700, 1000],
    #             'classifier__hidden_layer_sizes':[(20,), (50,), (100,), (150,), 
    #                                               (20,20),(50,50),(100,100), (150,150),
    #                                               (20,20,20),(50,50,50), (100,100,100),
    #                                               (150,150,150), (50,100,50)],
    #             'classifier__activation': ['relu'],
    #             'classifier__solver': ['adam'],
    #             'classifier__learning_rate': ["constant", "invscaling", "adaptive"],
    #             'classifier__alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]}
    # _scores = {'AUC': 'roc_auc', 'Recall': 'recall', 'Precision': 'precision'}
    # clf_MLP = RandomizedSearchCV(model_MLP, parameters, n_iter = 100,
    #                              scoring=_scores, refit='AUC', n_jobs=-1, cv=skf)
    # clf_MLP.fit(X_tr,y_tr)
    
    # clf_MLP_list.append(clf_MLP)
    
    parameters={'selector__k': [50, 70, 90, 100, 300, 700, 1000],
                'classifier__n_neighbors':[5, 10, 20, 50, 80, 110, 150, 200],
                'classifier__weights': ['uniform', 'distance']}
    _scores = {'AUC': 'roc_auc', 'Recall': 'recall', 'Precision': 'precision'}
    clf_knn = RandomizedSearchCV(model_knn, parameters, n_iter = 100,
                                 scoring=_scores, refit='AUC', n_jobs=-1, cv=skf)
    clf_knn.fit(X_tr,y_tr)
    
    clf_knn_list.append(clf_knn)

    parameters = {'selector__k': [10, 20, 50, 100, 150, 200, 300, 500, 700, 1000, 1500],
                  'classifier__kernel': ('rbf', 'poly', 'sigmoid'),
                  'classifier__degree': [2,3,4,5],
                  'classifier__C': [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 7, 10, 20, 50, 100],
                  'classifier__gamma': [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 7, 10, 20, 50, 100]
                }
    
    clf_SVC = GridSearchCV(model_SVC, parameters, scoring='accuracy', n_jobs=-1, cv=skf)
    clf_SVC.fit(X_tr,y_tr)
    clf_SVC_s.append(clf_SVC)
    
    parameters = {'selector__k': [10, 20, 50, 100, 150, 200, 300, 500, 700, 1000, 1500],
                'classifier__bootstrap': [True, False],
                'classifier__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                'classifier__max_features': ['auto', 'sqrt'],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
                'classifier__criterion': ('gini', 'entropy')}
    
    clf_RFC = GridSearchCV(model_RFC, parameters, scoring='accuracy', n_jobs=-1, cv=skf)
    clf_RFC.fit(X_tr,y_tr)
    clf_RFC_s.append(clf_RFC)
        
    parameters = {'selector__k': [150, 200, 500, 1000, 1500],
                  'classifier__hidden_layer_sizes':[(20,), (50,), (100,), (150,), 
                                                  (20,20),(50,50),(100,100), (150,150),
                                                  (20,20,20),(50,50,50), (100,100,100)],
                  'classifier__activation': ['relu'],
                  'classifier__solver': ['adam'],
                  'classifier__learning_rate': ['adaptive'],
                  'classifier__alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
                  }
    clf_MLP = GridSearchCV(model_MLP, parameters, scoring='accuracy', n_jobs=-1, cv=skf, verbose=10)
    clf_MLP.fit(X_tr,y_tr)
    clf_MLP_s.append(clf_MLP)


