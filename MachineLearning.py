from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from FeatureSelection import overall_best_fts
from sklearn.linear_model import LogisticRegression
import numpy as np

#%% SVM + Overall Best Feature Selection
def svm_overall_bst_fts(dataset, fts_names, labels_names, mode, scoring):
    
    X_train = dataset['X_tr']
    y_train = dataset['y_tr']
    
    best_fts = []
    validation_score = []
    best_estimators = []
    reduced_datasets = []
    
    # Feature Normalization
    norm_scaler = StandardScaler(with_mean=True, with_std=True)
    
    # SVC Model
    svc = SVC(C=0.1, gamma=0.01, kernel = 'rbf', random_state=42, probability=True)
    model_SVC = Pipeline(steps=[('norm_scaler',norm_scaler), ('classifier', svc)])
    
    # Cross-Validation
    skf = StratifiedKFold(n_splits=5)
    
    for train_index, test_index in skf.split(X_train, y_train):
        print(1)
        X_tr, X_val = X_train[train_index], X_train[test_index]
        y_tr, y_val = y_train[train_index], y_train[test_index]
        
        X_tr_pre_selected, X_val, reduced_dataset, best_fts_temp=overall_best_fts(X_tr,y_tr, X_val, X_train, y_train, fts_names,
                                                               estimator=model_SVC, mode=mode,
                                                               k_features_bdp=20,
                                                                 k_features_graph=150,
                                                                 k_features_asy=50,
                                                                 k_features_conn=50,
                                                                 n_features_to_select=25,
                                                                 scoring=scoring,
                                                                 cv=5)
        
        print(2)
        
        best_fts.append(best_fts_temp)
        reduced_datasets.append(reduced_dataset)
        
        space = dict({
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__gamma': [0.01, 0.1, 1, 10, 100],
        'classifier__kernel': ['rbf', 'sigmoid']
        })
        
        clf = GridSearchCV( estimator=model_SVC,
                            param_grid=space,
                            scoring=scoring, 
                            n_jobs=-1,
                            cv=skf )
        
        clf.fit(X_tr_pre_selected, y_tr)
        
        best_estimators.append(clf.best_estimator_)
        
        validation_score.append(clf.best_estimator_.score(X_val, y_val))
        
        
        print(3)
        
    mean_validation_score=np.mean(validation_score)
    std_validation_score=np.std(validation_score)
        
    return best_fts, best_estimators, validation_score, mean_validation_score, std_validation_score, reduced_datasets

#%% SVM + SelectKBest
def grid_search_svm_anova(dataset, labels_names):
    
    model = 'ANOVA + SVM'
    mode = dataset['MODE']
    scoring = dataset['SCORING']
    
    # Feature Normalization
    norm_scaler = StandardScaler(with_mean=True, with_std=True)
    
    # SVC Model
    svc = SVC(random_state=42)
    
    # Parameters for Grid Search
    space = dict({
        'classifier__C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5],
        'classifier__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 1],
        'classifier__kernel': ['rbf', 'linear', 'sigmoid']
    })
    
    # Feature Selection
    dim_red = SelectKBest(score_func=f_classif)
    
    space['dim_red__k'] = np.arange(5,30,1)
    
    # Pipeline
    model_SVC = Pipeline(steps=[('norm_scaler',norm_scaler),
                                ('dim_red', dim_red),
                                ('classifier', svc)])
    
    clf = GridSearchCV( estimator=model_SVC,
                        param_grid=space,
                        scoring=scoring, 
                        n_jobs=-1,
                        cv=5,
                        return_train_score=True )
    
    X_tr = dataset['X_tr']
    y_tr = dataset['y_tr']
    clf.fit(X_tr, y_tr)
    
    print('MODE:  ' + mode + '\nMODEL: ' + model)
    print('\nHYPERPARAMETERS')
    print(clf.best_params_, '\n')
    print('TRAIN SCORE')
    print(clf.cv_results_['mean_train_score'][clf.best_index_], '\n')
    print('VALIDATION SCORE')
    print(clf.best_score_, '\n')

    return clf.best_params_, model, clf

def svm_anova_estimators(dataset, gs_svm_anova, model):
    
    model = 'SVM & ANOVA'
    
    pipe = Pipeline(steps=[('norm_scaler', StandardScaler(with_mean=True, with_std=True)),
                            ('dim_red', SelectKBest(score_func=f_classif)),
                            ('classifier', SVC(random_state=42, probability=True))])
    
    pipe.set_params(**gs_svm_anova)
    
    scores_pipe = cross_validate(   estimator=pipe,
                                    X=dataset['X_tr'],
                                    y=dataset['y_tr'],
                                    scoring=['roc_auc'],
                                    cv=5,
                                    return_train_score=True,
                                    return_estimator=True)
    
    return scores_pipe['estimator']

#%% SVM + PCA

def grid_search_svm_pca(dataset, labels_names):

    model = 'SVM + PCA'
    mode = dataset['MODE']
    scoring = dataset['SCORING']    
    
    # Feature Normalization
    norm_scaler = StandardScaler(with_mean=True, with_std=True)
    
    # SVC Model
    svc = SVC(random_state=42)
    
    # Parameters for Grid Search
    space = dict({
        'classifier__C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5],
        'classifier__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 1],
        'classifier__kernel': ['rbf', 'linear', 'sigmoid']
    })
    
    # Dimensionality Reduction
    dim_red = PCA(random_state=42)
    
    space['dim_red__n_components'] = np.arange(3,20,1)
    
    # Pipeline
    model_SVC = Pipeline(steps=[
                                ('norm_scaler', norm_scaler),
                                ('dim_red', dim_red),
                                ('classifier', svc)])
    
    clf = GridSearchCV( estimator=model_SVC,
                        param_grid=space,
                        scoring=scoring, 
                        n_jobs=-1,
                        cv=5,
                        return_train_score=True )
    
    X_tr = dataset['X_tr']
    y_tr = dataset['y_tr']
    clf.fit(X_tr, y_tr)
   
    print('MODE:  ' + mode + '\nMODEL: ' + model)
    print('\nHYPERPARAMETERS')
    print(clf.best_params_, '\n')
    print('TRAIN SCORE')
    print(clf.cv_results_['mean_train_score'][clf.best_index_], '\n')
    print('VALIDATION SCORE')
    print(clf.best_score_, '\n')

    return clf.best_params_, model, clf

def svm_pca_estimators(dataset, gs_svm_pca, model):
    
    model = 'SVM & PCA'
    
    pipe = Pipeline(steps=[('norm_scaler', StandardScaler(with_mean=True, with_std=True)),
                            ('dim_red', PCA(random_state=42)),
                            ('classifier', SVC(random_state=42, probability=True))])
    
    pipe.set_params(**gs_svm_pca)
    
    scores_pipe = cross_validate(   estimator=pipe,
                                    X=dataset['X_tr'],
                                    y=dataset['y_tr'],
                                    scoring=['roc_auc'],
                                    cv=5,
                                    return_train_score=True,
                                    return_estimator=True)
    
    return scores_pipe['estimator']

#%% MLP + SelectKBest

def grid_search_mlp_anova(dataset, labels_names):
    
    model = 'MLP + ANOVA'
    mode = dataset['MODE']
    scoring = dataset['SCORING']
    
    # Feature Normalization
    norm_scaler = StandardScaler(with_mean=True, with_std=True)
    minMax_scaler = MinMaxScaler()
    
    # MLP Model
    mlp = MLPClassifier(random_state=42,
                        max_iter=500,
                        early_stopping=True,
                        activation='relu',
                        solver='adam')
    
    # Parameters for Grid Search
    space = dict({
        'classifier__hidden_layer_sizes':[(50),(75),(100),(125),(150),(175),(200),(250),(300),(500), 
                                          (100,100),(125,125),(150,150),(175,175),(200,200),
                                          (100,100,100), (150,150,150),(200,200,200)],
        'classifier__alpha':[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
        'classifier__learning_rate': ['adaptive']
    })
    
    # Feature Selection
    dim_red = SelectKBest(score_func=f_classif)
    
    space['dim_red__k'] = np.arange(5,30,1)
    
    # Pipeline
    model_MLP = Pipeline(steps=[('norm_scaler',norm_scaler),
                                ('min_max', minMax_scaler),
                                ('dim_red', dim_red),
                                ('classifier', mlp)])
    
    clf = GridSearchCV( estimator=model_MLP,
                        param_grid=space,
                        scoring=scoring, 
                        n_jobs=-1,
                        cv=5,
                        return_train_score=True )
    
    X_tr = dataset['X_tr']
    y_tr = dataset['y_tr']
    clf.fit(X_tr, y_tr)
    
    print('MODE:  ' + mode + '\nMODEL: ' + model)
    print('\nHYPERPARAMETERS')
    print(clf.best_params_, '\n')
    print('TRAIN SCORE')
    print(clf.cv_results_['mean_train_score'][clf.best_index_], '\n')
    print('VALIDATION SCORE')
    print(clf.best_score_, '\n')
    
    return clf.best_params_, model, clf

def mlp_anova_estimators(dataset, gs_mlp_anova, model):
    
    model = 'MLP & ANOVA'
    
    pipe = Pipeline(steps=[('norm_scaler', StandardScaler(with_mean=True, with_std=True)),
                            ('min_max', MinMaxScaler()),
                            ('dim_red', SelectKBest(score_func=f_classif)),
                            ('classifier', MLPClassifier(random_state=42, max_iter=500, early_stopping=True, activation='relu', solver='adam'))])
    
    pipe.set_params(**gs_mlp_anova)
    
    scores_pipe = cross_validate(   estimator=pipe,
                                    X=dataset['X_tr'],
                                    y=dataset['y_tr'],
                                    scoring=['roc_auc'],
                                    cv=5,
                                    return_train_score=True,
                                    return_estimator=True)
    
    return scores_pipe['estimator']

#%% MLP + PCA

def grid_search_mlp_pca(dataset, labels_names):

    model = 'MLP + PCA'
    mode = dataset['MODE']
    scoring = dataset['SCORING']
    
    # Feature Normalization
    norm_scaler = StandardScaler(with_mean=True, with_std=True)
    minMax_scaler = MinMaxScaler()
    
    # MLP Model
    mlp = MLPClassifier(random_state=42,
                        max_iter=500,
                        early_stopping=True,
                        activation='relu',
                        solver='adam',
                        learning_rate='adaptive')
    
    # Parameters for Grid Search
    space = dict({
        'classifier__hidden_layer_sizes':[(50),(75),(100),(125),(150),(175),(200),(250),(300),(500), 
                                          (100,100),(125,125),(150,150),(175,175),(200,200),
                                          (100,100,100), (150,150,150),(200,200,200)],
        'classifier__alpha':[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
        'classifier__learning_rate': ['adaptive']
    })
    
    # Dimensionality Reduction
    dim_red = PCA(random_state=42)
    
    space['dim_red__n_components'] = np.arange(3,20,1)
    
    
    # Pipeline
    model_MLP = Pipeline(steps=[('norm_scaler',norm_scaler),
                                ('min_max', minMax_scaler),
                                ('dim_red', dim_red),
                                ('classifier', mlp)])
    
    clf = GridSearchCV( estimator=model_MLP,
                        param_grid=space,
                        scoring=scoring, 
                        n_jobs=-1,
                        cv=5,
                        return_train_score=True )
    
    X_tr = dataset['X_tr']
    y_tr = dataset['y_tr']
    clf.fit(X_tr, y_tr)
    
    print('MODE:  ' + mode + '\nMODEL: ' + model)
    print('\nHYPERPARAMETERS')
    print(clf.best_params_, '\n')
    print('TRAIN SCORE')
    print(clf.cv_results_['mean_train_score'][clf.best_index_], '\n')
    print('VALIDATION SCORE')
    print(clf.best_score_, '\n')
    
    return clf.best_params_, model, clf

def mlp_pca_estimators(dataset, gs_mlp_pca, model):
    
    model = 'MLP & PCA'
    
    pipe = Pipeline(steps=[('norm_scaler', StandardScaler(with_mean=True, with_std=True)),
                            ('min_max', MinMaxScaler()),
                            ('dim_red', PCA(random_state=42)),
                            ('classifier', MLPClassifier(random_state=42, max_iter=500, early_stopping=True, activation='relu', solver='adam'))])
    
    pipe.set_params(**gs_mlp_pca)
    
    scores_pipe = cross_validate(   estimator=pipe,
                                    X=dataset['X_tr'],
                                    y=dataset['y_tr'],
                                    scoring=['roc_auc'],
                                    cv=5,
                                    return_train_score=True,
                                    return_estimator=True)
    
    return scores_pipe['estimator']

#%% RFC + SelectKBest

def grid_search_rfc_anova(dataset, labels_names):
    
    model = 'RFC + ANOVA'
    mode = dataset['MODE']
    scoring = dataset['SCORING']
    
    
    # Feature Normalization
    norm_scaler = StandardScaler(with_mean=True, with_std=True)
    
    # RFC Model
    rfc = RandomForestClassifier(random_state=42)
    
    # Parameters for Grid Search
    space = dict({
        'classifier__bootstrap': [True],
        'classifier__max_depth': [5, 10, 25, 50, 90, None],
        'classifier__max_features': [None],
        'classifier__min_samples_leaf': [1, 5],
        'classifier__min_samples_split': [2, 5],
        'classifier__n_estimators': [50, 75, 100, 150, 250, 500],
        'classifier__criterion': ['gini']
    })
    
    # Feature Selection
    dim_red = SelectKBest(score_func=f_classif)
    
    space['dim_red__k'] = np.arange(5,30,1)
    
    # Pipeline
    model_RFC = Pipeline(steps=[('norm_scaler',norm_scaler),
                                ('dim_red', dim_red),
                                ('classifier', rfc)])
    
    clf = GridSearchCV( estimator=model_RFC,
                        param_grid=space,
                        scoring=scoring, 
                        n_jobs=-1,
                        cv=5,
                        return_train_score=True )
    
    X_tr = dataset['X_tr']
    y_tr = dataset['y_tr']
    clf.fit(X_tr, y_tr)
    
    print('MODE:  ' + mode + '\nMODEL: ' + model)
    print('\nHYPERPARAMETERS')
    print(clf.best_params_, '\n')
    print('TRAIN SCORE')
    print(clf.cv_results_['mean_train_score'][clf.best_index_], '\n')
    print('VALIDATION SCORE')
    print(clf.best_score_, '\n')
    
    return clf.best_params_, model, clf

def rfc_anova_estimators(dataset, gs_rfc_anova, model):
    
    pipe = Pipeline(steps=[('norm_scaler', StandardScaler(with_mean=True, with_std=True)),
                            ('dim_red', SelectKBest(score_func=f_classif)),
                            ('classifier', RandomForestClassifier(random_state=42))])
    
    pipe.set_params(**gs_rfc_anova)
    
    scores_pipe = cross_validate(   estimator=pipe,
                                    X=dataset['X_tr'],
                                    y=dataset['y_tr'],
                                    scoring=['roc_auc'],
                                    cv=5,
                                    return_train_score=True,
                                    return_estimator=True)
    
    return scores_pipe['estimator']

#%% RFC + Built-in Feature Selection

def grid_search_rfc(dataset, labels_names):
    
    model = 'RFC + BUILT-IN FEATURE SELECTION'
    mode = dataset['MODE']
    scoring = dataset['SCORING']
    
    
    # Feature Normalization
    norm_scaler = StandardScaler(with_mean=True, with_std=True)
    
    # RFC Model
    rfc = RandomForestClassifier(random_state=42)
    
    # Parameters for Grid Search
    # space = dict({
    #     'classifier__bootstrap': [True],
    #     'classifier__max_depth': [5, 10, 25, 50, None],
    #     'classifier__max_features': np.arange(3,20,1),
    #     'classifier__min_samples_leaf': [1, 5],
    #     'classifier__min_samples_split': [2, 5],
    #     'classifier__n_estimators': [175, 185, 200, 220, 275],
    #     'classifier__criterion': ['gini']
    # })
    
    space = dict({
        'classifier__bootstrap': [True],
        'classifier__max_depth': [5],
        'classifier__max_features': [5],
        'classifier__min_samples_leaf': [1],
        'classifier__min_samples_split': [2],
        'classifier__n_estimators': [220],
        'classifier__criterion': ['gini']
    })
    
    # Pipeline
    model_RFC = Pipeline(steps=[('norm_scaler',norm_scaler),
                                ('classifier', rfc)])
    
    clf = GridSearchCV( estimator=model_RFC,
                        param_grid=space,
                        scoring=scoring, 
                        n_jobs=-1,
                        cv=5,
                        return_train_score=True )
    
    X_tr = dataset['X_tr']
    y_tr = dataset['y_tr']
    clf.fit(X_tr, y_tr)
    
    print('MODE:  ' + mode + '\nMODEL: ' + model)
    print('\nHYPERPARAMETERS')
    print(clf.best_params_, '\n')
    print('TRAIN SCORE')
    print(clf.cv_results_['mean_train_score'][clf.best_index_], '\n')
    print('VALIDATION SCORE')
    print(clf.best_score_, '\n')
    
    return clf.best_params_, model, clf

def rfc_estimators(dataset, gs_rfc, model):
    
    pipe = Pipeline(steps=[('norm_scaler', StandardScaler(with_mean=True, with_std=True)),
                            ('classifier', RandomForestClassifier(random_state=42))])
    
    pipe.set_params(**gs_rfc)
    
    scores_pipe = cross_validate(   estimator=pipe,
                                    X=dataset['X_tr'],
                                    y=dataset['y_tr'],
                                    scoring=['roc_auc'],
                                    cv=5,
                                    return_train_score=True,
                                    return_estimator=True)
    
    return scores_pipe['estimator']


#%% RFC + PCA

def grid_search_rfc_pca(dataset, labels_names):
    
    model = 'RFC + PCA'
    mode = dataset['MODE']
    scoring = dataset['SCORING']
    
    # Feature Normalization
    norm_scaler = StandardScaler(with_mean=True, with_std=True)
    
    # RFC Model
    rfc = RandomForestClassifier(random_state=42)
    
    # Parameters for Grid Search
    space = dict({
        'classifier__bootstrap': [True],
        'classifier__max_depth': [5, 10, 25, 50, 90, None],
        'classifier__max_features': [None],
        'classifier__min_samples_leaf': [1, 5],
        'classifier__min_samples_split': [2, 5],
        'classifier__n_estimators': [50, 75, 100, 150, 250, 500],
        'classifier__criterion': ['gini']
    })
    
    # Dimensionality Reduction
    dim_red = PCA(random_state=42)
    
    space['dim_red__n_components'] = np.arange(3,20,1)
    
    # Pipeline
    model_RFC = Pipeline(steps=[('norm_scaler',norm_scaler),
                                ('dim_red', dim_red),
                                ('classifier', rfc)])
    
    clf = GridSearchCV( estimator=model_RFC,
                        param_grid=space,
                        scoring=scoring, 
                        n_jobs=-1,
                        cv=5,
                        return_train_score=True )
    
    X_tr = dataset['X_tr']
    y_tr = dataset['y_tr']
    clf.fit(X_tr, y_tr)
    
    print('MODE:  ' + mode + '\nMODEL: ' + model)
    print('\nHYPERPARAMETERS')
    print(clf.best_params_, '\n')
    print('TRAIN SCORE')
    print(clf.cv_results_['mean_train_score'][clf.best_index_], '\n')
    print('VALIDATION SCORE')
    print(clf.best_score_, '\n')
    
    return clf.best_params_, model, clf

def rfc_pca_estimators(dataset, gs_rfc_pca, model):
    
    pipe = Pipeline(steps=[('norm_scaler', StandardScaler(with_mean=True, with_std=True)),
                            ('dim_red', PCA(random_state=42)),
                            ('classifier', RandomForestClassifier(random_state=42))])
    
    pipe.set_params(**gs_rfc_pca)
    
    scores_pipe = cross_validate(   estimator=pipe,
                                    X=dataset['X_tr'],
                                    y=dataset['y_tr'],
                                    scoring=['roc_auc'],
                                    cv=5,
                                    return_train_score=True,
                                    return_estimator=True)
    
    return scores_pipe['estimator']

#%% LogReg + SelectKBest

def grid_search_logReg_anova(dataset, labels_names):
    
    model = 'LogReg + ANOVA'
    mode = dataset['MODE']
    scoring = dataset['SCORING']
    
    
    # Feature Normalization
    norm_scaler = StandardScaler(with_mean=True, with_std=True)
    
    # RFC Model
    logReg = LogisticRegression(random_state=42, max_iter=300)
    
    # Parameters for Grid Search
    space = dict({
        'classifier__C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 10],
        'classifier__class_weight': ['balanced', None],
        'classifier__solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    })
    
    # Feature Selection
    dim_red = SelectKBest(score_func=f_classif)
    
    space['dim_red__k'] = np.arange(5,30,1)
    
    # Pipeline
    model_logReg = Pipeline(steps=[('norm_scaler', norm_scaler),
                                ('dim_red', dim_red),
                                ('classifier', logReg)])
    
    clf = GridSearchCV( estimator=model_logReg,
                        param_grid=space,
                        scoring=scoring, 
                        n_jobs=-1,
                        cv=5,
                        return_train_score=True )
    
    X_tr = dataset['X_tr']
    y_tr = dataset['y_tr']
    clf.fit(X_tr, y_tr)
    
    print('MODE:  ' + mode + '\nMODEL: ' + model)
    print('\nHYPERPARAMETERS')
    print(clf.best_params_, '\n')
    print('TRAIN SCORE')
    print(clf.cv_results_['mean_train_score'][clf.best_index_], '\n')
    print('VALIDATION SCORE')
    print(clf.best_score_, '\n')
    
    return clf.best_params_, model, clf

def logReg_anova_estimators(dataset, gs_logReg_anova, model):
    
    pipe = Pipeline(steps=[('norm_scaler', StandardScaler(with_mean=True, with_std=True)),
                            ('dim_red', SelectKBest(score_func=f_classif)),
                            ('classifier', LogisticRegression(random_state=42, max_iter=300))])
    
    pipe.set_params(**gs_logReg_anova)
    
    scores_pipe = cross_validate(   estimator=pipe,
                                    X=dataset['X_tr'],
                                    y=dataset['y_tr'],
                                    scoring=['roc_auc'],
                                    cv=5,
                                    return_train_score=True,
                                    return_estimator=True)
    
    return scores_pipe['estimator']

#%% LogReg + PCA

def grid_search_logReg_pca(dataset, labels_names):
    
    model = 'LogReg + PCA'
    mode = dataset['MODE']
    scoring = dataset['SCORING']
    
    
    # Feature Normalization
    norm_scaler = StandardScaler(with_mean=True, with_std=True)
    
    # RFC Model
    logReg = LogisticRegression(random_state=42)
    
    # Parameters for Grid Search
    space = dict({
        'classifier__C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 10],
        'classifier__class_weight': ['balanced', None],
        'classifier__solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    })
    
    # Feature Selection
    dim_red = PCA(random_state=42)
    
    space['dim_red__n_components'] = np.arange(3,20,1)
    
    # Pipeline
    model_logReg = Pipeline(steps=[('norm_scaler',norm_scaler),
                                ('dim_red', dim_red),
                                ('classifier', logReg)])
    
    clf = GridSearchCV( estimator=model_logReg,
                        param_grid=space,
                        scoring=scoring, 
                        n_jobs=-1,
                        cv=5,
                        return_train_score=True )
    
    X_tr = dataset['X_tr']
    y_tr = dataset['y_tr']
    clf.fit(X_tr, y_tr)
    
    print('MODE:  ' + mode + '\nMODEL: ' + model)
    print('\nHYPERPARAMETERS')
    print(clf.best_params_, '\n')
    print('TRAIN SCORE')
    print(clf.cv_results_['mean_train_score'][clf.best_index_], '\n')
    print('VALIDATION SCORE')
    print(clf.best_score_, '\n')
    
    return clf.best_params_, model, clf

def logReg_pca_estimators(dataset, gs_logReg_pca, model):
    
    pipe = Pipeline(steps=[('norm_scaler', StandardScaler(with_mean=True, with_std=True)),
                            ('dim_red', PCA(random_state=42)),
                            ('classifier', LogisticRegression(random_state=42))])
    
    pipe.set_params(**gs_logReg_pca)
    
    scores_pipe = cross_validate(   estimator=pipe,
                                    X=dataset['X_tr'],
                                    y=dataset['y_tr'],
                                    scoring=['roc_auc'],
                                    cv=5,
                                    return_train_score=True,
                                    return_estimator=True)
    
    return scores_pipe['estimator']