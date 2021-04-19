from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV



#%% SVM + SelectKBest
def svm_anova(dataset, mode, scoring):

    print(mode + ' SVM + SelectKBest\n')
    
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
                        scoring=scoring, 
                        n_jobs=-1,
                        cv=skf,
                        refit=False,
                        return_train_score=True )
    
    X_tr = dataset['X_tr']
    y_tr = dataset['y_tr']
    clf.fit(X_tr, y_tr)
    print(scoring + ' :')
    print(clf.best_score_)
    print('best parameters: ')
    print(clf.best_params_)

    return clf

#%% SVM + PCA

def svm_pca(dataset, mode, scoring):

    print(mode + ' SVM + PCA\n')
    
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
    print(scoring + ' :')
    print(clf.best_score_)
    print('best parameters: ')
    print(clf.best_params_)
    
    return clf

#%% MLP + SelectKBest

def mlp_anova(dataset, mode, scoring):
    
    print(mode + ' MLP + SelectKBest\n')
    
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
    print(scoring + ' :')
    print(clf.best_score_)
    print('best parameters: ')
    print(clf.best_params_)
    
    return clf

#%% MLP + PCA

def mlp_pca(dataset, mode, scoring):

    print(mode + ' MLP + PCA\n')
    
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
    print(scoring + ' :')
    print(clf.best_score_)
    print('best parameters: ')
    print(clf.best_params_)

    return clf

#%% RFC + SelectKBest

def rfc_anova(dataset, mode, scoring):

    print(mode + ' RFC + SelectKBest\n')
    
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
    print(scoring + ' :')
    print(clf.best_score_)
    print('best parameters: ')
    print(clf.best_params_)
    
    return clf


#%% RFC + PCA

def rfc_pca(dataset, mode, scoring):

    print(mode + ' RFC + PCA\n')
    
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
    print(scoring + ' :')
    print(clf.best_score_)
    print('best parameters: ')
    print(clf.best_params_)
    
    return clf
