from Pickle import getPickleFile
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
# from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV
# import seaborn as sb
# import pandas as pd

datasets = getPickleFile('../ML_Data/' + 'datasets')

for dset in datasets[0:1]:

    X_tr = dset['X_tr']
    y_tr = dset['y_tr']
   
    #Feature Normalization
    norm_scaler = StandardScaler(with_mean=True, with_std=True)
    minMax_scaler = MinMaxScaler()

    # #Feature Selection
    # _k_features = 200
    
    selector = SelectKBest(score_func=f_classif)
    
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
    mlp = MLPClassifier(random_state=42, max_iter = 1000, early_stopping = True, verbose = True)
    model_MLP = Pipeline(steps=[('norm_scaler',norm_scaler),
                                ('minMax_scaler', minMax_scaler),
                                ('selector', selector),
                                ('classifier', mlp)])
    
    # Cross-Validation
    skf = StratifiedKFold(n_splits=5)

    #GridSearchCV
    parameters={'selector__k': [10, 20, 50, 100, 150, 200, 300, 500, 700, 1000, 1500],
                'classifier__kernel': ('rbf', 'poly', 'sigmoid'),
                'classifier__degree': [2,3,4,5],
                'classifier__C': [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 7, 10, 20, 50, 100],
                'classifier__gamma': [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 7, 10, 20, 50, 100]}
    
    clf_SVC = GridSearchCV(model_SVC, parameters, scoring='accuracy', n_jobs=-1, cv=skf)
    clf_SVC.fit(X_tr,y_tr)
    
    parameters={'selector__k': [10, 20, 50, 100, 150, 200, 300, 500, 700, 1000, 1500],
                'classifier__bootstrap': [True, False],
                'classifier__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                'classifier__max_features': ['auto', 'sqrt'],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
                'classifier__criterion': ('gini', 'entropy')}
    
    clf_RFC = GridSearchCV(model_RFC, parameters, scoring='accuracy', n_jobs=-1, cv=skf)
    clf_RFC.fit(X_tr,y_tr)
        
    parameters={'selector__k': [10, 20, 50, 100, 150, 200, 300, 500, 700, 1000, 1500],
                'classifier__hidden_layer_sizes':[(20,), (50,), (100,), (150,), 
                                                  (20,20),(50,50),(100,100), (150,150),
                                                  (20,20,20),(50,50,50), (100,100,100),
                                                  (150,150,150), (50,100,50)],
                'classifier__activation': ('relu'),
                'classifier__solver': ('adam'),
                'classifier__learning_rate': ["constant", "invscaling", "adaptive"],
                'classifier__alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]}
    clf_MLP = GridSearchCV(model_MLP, parameters, scoring='accuracy', n_jobs=-1, cv=skf)
    clf_MLP.fit(X_tr,y_tr)
    