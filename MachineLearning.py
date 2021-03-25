from Pickle import getPickleFile
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
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
    
    # SVC
    _C = 1.5
    _kernel = 'rbf'
    
    SVC_classifier = SVC(random_state=42, C=_C, kernel=_kernel)
    
    #RFC
    _criterion = 'gini'
    _n_estimators = 40
    RFC_classifier = RandomForestClassifier(random_state=42,
                                            criterion = _criterion,
                                            n_estimators = _n_estimators)
    
    #NN
    _activation = 'logistic'
    _hidden_layer_sizes = 60
    _max_iter = 600
    MLP_classifier = MLPClassifier(random_state=42, activation = _activation,
                                    hidden_layer_sizes = _hidden_layer_sizes,
                                    max_iter = _max_iter)
    
    # Cross-Validation
    skf = StratifiedKFold(n_splits=5)
    
    # models
    model_SVC = make_pipeline(norm_scaler, minMax_scaler, selector,
                              SVC_classifier)
    model_RFC = make_pipeline(norm_scaler, minMax_scaler, selector,
                              RFC_classifier)
    model_MLP = make_pipeline(norm_scaler, minMax_scaler, selector,
                              MLP_classifier)
    
    # Scores
    score_SVC = cross_validate(model_SVC, X_tr, y_tr, cv=skf,
                            scoring=('accuracy', 'balanced_accuracy', 'roc_auc'),
                            return_train_score=True)
    
    score_RFC = cross_validate(model_RFC, X_tr, y_tr, cv=skf,
                            scoring=('accuracy', 'balanced_accuracy', 'roc_auc'),
                            return_train_score=True)
    
    score_MLP = cross_validate(model_MLP, X_tr, y_tr, cv=skf,
                            scoring=('accuracy', 'balanced_accuracy', 'roc_auc'),
                            return_train_score=True)
    