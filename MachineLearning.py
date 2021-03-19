from Pickle import getPickleFile
# from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate


datasets = getPickleFile('../ML_Data/' + 'datasets')

for dset in datasets:

    X_tr = dset['X_tr']
    y_tr = dset['y_tr']
   
    # Feature Normalization
    # norm_scaler = StandardScaler(with_mean=True, with_std=True)
    
    # X_tr = norm_scaler.fit_transform(X_tr)

    # Feature Selection
    _k_features = 200
    
    selector = SelectKBest(score_func=f_classif,
                           k=_k_features)
    
    # Classifier
    _C = 0.1
    _kernel = 'rbf'
    _gamma = 'scale'
    
    classifier = SVC(random_state=42, 
                      C=_C, 
                      kernel=_kernel, 
                      gamma=_gamma)
    
    # Selection + Classifier
    model = make_pipeline(selector, classifier)
    
    # Cross-Validation
    skf = StratifiedKFold(n_splits=5)
    
    scores = cross_validate(model, X_tr, y_tr, cv=skf,
                            scoring=('accuracy', 'balanced_accuracy', 'roc_auc'),
                            return_train_score=True)
    