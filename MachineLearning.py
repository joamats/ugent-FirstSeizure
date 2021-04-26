from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate

#%% SVM + SelectKBest
def grid_search_svm_anova(dataset, labels_names):
    
    model = 'ANOVA + SVM'
    mode = dataset['MODE']
    scoring = dataset['SCORING']
    
    # Feature Normalization
    norm_scaler = StandardScaler(with_mean=True, with_std=True)
    
    # SVC Model
    svc = SVC(random_state=42)
    
    # Cross-Validation
    skf = StratifiedKFold(n_splits=5)
    
    # Parameters for Grid Search
    space = dict({
        'classifier__C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10],
        'classifier__gamma': [0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 100],
        'classifier__kernel': ['rbf', 'sigmoid']
    })
    
    # Feature Selection
    dim_red = SelectKBest(score_func=f_classif)
    
    space['dim_red__k'] = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    
    # Pipeline
    model_SVC = Pipeline(steps=[('norm_scaler',norm_scaler),
                                ('dim_red', dim_red),
                                ('classifier', svc)])
    
    clf = GridSearchCV( estimator=model_SVC,
                        param_grid=space,
                        scoring=scoring, 
                        n_jobs=-1,
                        cv=skf,
                        return_train_score=True )
    
    X_tr = dataset['X_tr']
    y_tr = dataset['y_tr']
    clf.fit(X_tr, y_tr)
    
    print('MODE:  ' + mode + '\nMODEL: ' + model)
    print('\nHYPERPARAMETERS:')
    print(clf.best_params_, '\n')
    
    return clf, model

def svm_anova_estimators(dataset, clf_svm_anova, model):
    
    model = 'SVM & ANOVA'

    clf_params = clf_svm_anova.get_params()
    
    pipe = Pipeline(steps=[('norm_scaler', StandardScaler(with_mean=True, with_std=True)),
                            ('dim_red', SelectKBest(score_func=f_classif)),
                            ('classifier', SVC(random_state=42, probability=True))])
    
    pipe_params = { 'dim_red__k': clf_params['estimator__dim_red__k'],
                    'classifier__C': clf_params['estimator__classifier__C'],
                    'classifier__gamma': clf_params['estimator__classifier__gamma'],
                    'classifier__kernel': clf_params['estimator__classifier__kernel'] }
    
    pipe.set_params(**pipe_params)
    
    X_tr = dataset['X_tr']
    y_tr = dataset['y_tr']
    
    dummy = DummyClassifier(strategy="most_frequent")
    
    skf = StratifiedKFold(n_splits=5)
    
    scores_dummy = cross_validate(  estimator=dummy,
                                    X=X_tr,
                                    y=y_tr,
                                    scoring=['roc_auc', 'f1_macro'],
                                    cv=skf,
                                    return_train_score=True,
                                    return_estimator=True)
    
    print('DUMMY CLASSIFIER')
    print('AUC ROC: {:.3f}'.format(scores_dummy['test_roc_auc'].mean()))
    print('Balanced F1-score: {:.3f}\n'.format(scores_dummy['test_f1_macro'].mean()))
    
    scores_pipe = cross_validate(   estimator=pipe,
                                    X=X_tr,
                                    y=y_tr,
                                    scoring=['roc_auc', 'f1_macro'],
                                    cv=skf,
                                    return_train_score=True,
                                    return_estimator=True)
    
    print(model + ' CLASSIFIER')
    print('AUC ROC: {:.3f}'.format(scores_pipe['test_roc_auc'].mean()))
    print('Balanced F1-score: {:.3f}'.format(scores_pipe['test_f1_macro'].mean()))
    
    estimators = scores_pipe['estimator']

    return estimators

#%% SVM + PCA

def svm_pca(dataset, labels_names, mode, scoring):

    model = 'SVM + PCA'
    
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
                        scoring=scoring, 
                        n_jobs=-1,
                        cv=skf,
                        return_train_score=True )
    
    X_tr = dataset['X_tr']
    y_tr = dataset['y_tr']
    clf.fit(X_tr, y_tr)
   
    return clf

#%% MLP + SelectKBest

def mlp_anova(dataset, labels_names, mode, scoring):
    
    model = 'MLP + ANOVA'
    
    # Feature Normalization
    norm_scaler = StandardScaler(with_mean=True, with_std=True)
    minMax_scaler = MinMaxScaler()
    
    # MLP Model
    mlp = MLPClassifier(random_state=42, max_iter = 1000, early_stopping = True)
    
    # Cross-Validation
    skf = StratifiedKFold(n_splits=5)
    
    # Parameters for Grid Search
    space = dict({
        'classifier__hidden_layer_sizes':[(100), (150), (200), (500), 
                                          (100,100), (150,150),(200,200), (500,500),
                                          (100,100,100),(150,150,150), (200,200,200)],
        'classifier__activation': ['relu'],
        'classifier__solver': ['adam'],
        'classifier__learning_rate': ['adaptive'],
        'classifier__alpha':[0.001, 0.01, 0.1, 1],
        'classifier__early_stopping': [False]
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
                        scoring=scoring, 
                        n_jobs=-1,
                        cv=skf,
                        return_train_score=True )
    
    X_tr = dataset['X_tr']
    y_tr = dataset['y_tr']
    clf.fit(X_tr, y_tr)
    
    return clf

#%% MLP + PCA

def mlp_pca(dataset, labels_names, mode, scoring):

    model = 'MLP + PCA'
    
    # Feature Normalization
    norm_scaler = StandardScaler(with_mean=True, with_std=True)
    minMax_scaler = MinMaxScaler()
    
    # MLP Model
    mlp = MLPClassifier(random_state=42, max_iter = 1000)
    
    # Cross-Validation
    skf = StratifiedKFold(n_splits=5)
    
    # Parameters for Grid Search
    space = dict({
        'classifier__hidden_layer_sizes':[(100), (150), (200), (500), 
                                          (100,100), (150,150),(200,200), (500,500),
                                          (100,100,100),(150,150,150), (200,200,200)],
        'classifier__activation': ['relu'],
        'classifier__solver': ['adam'],
        'classifier__learning_rate': ['adaptive'],
        'classifier__alpha':[0.001, 0.01, 0.1, 1],
        'classifier__early_stopping': [False]
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
                        scoring=scoring, 
                        n_jobs=-1,
                        cv=skf,
                        return_train_score=True )
    
    X_tr = dataset['X_tr']
    y_tr = dataset['y_tr']
    clf.fit(X_tr, y_tr)
    
    return clf

#%% RFC + SelectKBest

def rfc_anova(dataset, labels_names, mode, scoring):

    model = 'RFC + ANOVA'
    
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
                        scoring=scoring, 
                        n_jobs=-1,
                        cv=skf,
                        return_train_score=True )
    
    X_tr = dataset['X_tr']
    y_tr = dataset['y_tr']
    clf.fit(X_tr, y_tr)
    
    return clf


#%% RFC + PCA

def rfc_pca(dataset, labels_names, mode, scoring):

    model = 'RFC + PCA'
    
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
                        scoring=scoring, 
                        n_jobs=-1,
                        cv=skf,
                        return_train_score=True )
    
    X_tr = dataset['X_tr']
    y_tr = dataset['y_tr']
    clf.fit(X_tr, y_tr)
    
    return clf  

    