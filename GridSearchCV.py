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

    #Feature Selection
    _k_features = 200
    
    selector = SelectKBest(score_func=f_classif,
                            k=_k_features)
    
     # models
    svc = SVC(random_state=42)
    model_SVC = Pipeline(steps=[('norm_scaler',norm_scaler),
                                ('minMax_scaler', minMax_scaler),
                                ('selector', selector),
                                ('classifier', svc)])
    rfc = RandomForestClassifier(random_state=42)
    model_RFC = Pipeline(steps=[('norm_scaler',norm_scaler),
                                ('minMax_scaler', minMax_scaler),
                                ('selector', selector),
                                ('classifier', rfc)])
    mlp = MLPClassifier(random_state=42, max_iter = 700)
    model_MLP = Pipeline(steps=[('norm_scaler',norm_scaler),
                                ('minMax_scaler', minMax_scaler),
                                ('selector', selector),
                                ('classifier', mlp)])
    
    # Cross-Validation
    skf = StratifiedKFold(n_splits=5)

    #GridSearchCV
    parameters={'classifier__kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
                'classifier__C':np.arange(1,10, 0.5).tolist()}
    
    clf_SVC = GridSearchCV(model_SVC, parameters, scoring='accuracy', cv=skf)
    clf_SVC.fit(X_tr,y_tr)
    
    parameters={'classifier__n_estimators':np.arange(10,200, 5).tolist(),
            'classifier__criterion': ('gini', 'entropy')}
    clf_RFC = GridSearchCV(model_RFC, parameters, scoring='accuracy', cv=skf)
    clf_RFC.fit(X_tr,y_tr)
        
    parameters={'classifier__hidden_layer_sizes':np.arange(20,200, 10).tolist(),
                'classifier__activation': ('logistic', 'relu'),
                'classifier__solver': ('adam', 'lbfgs'),
                'classifier__alpha':np.arange(0.0001,1, 0.1).tolist()}
    clf_MLP = GridSearchCV(model_MLP, parameters, scoring='accuracy', cv=skf)
    clf_MLP.fit(X_tr,y_tr)
    