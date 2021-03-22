from Pickle import getPickleFile
# from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.manifold import TSNE

import seaborn as sb
import pandas as pd

datasets = getPickleFile('../ML_Data/' + 'datasets')

for dset in datasets[0:1]:

    X_tr = dset['X_tr']
    y_tr = dset['y_tr']
   
    X_embedded = TSNE(n_components=2).fit_transform(X_tr)
     
    df = pd.DataFrame()
    df['one'] = X_embedded[:,0]
    df['two'] = X_embedded[:,1]
    df['y'] = y_tr
    
    sb.scatterplot(
        x="one", y="two",
        hue="y",
        palette=sb.color_palette("hls", 2),
        data=df,
        legend="full",
        alpha=0.8
    )
   
    # Feature Normalization
    # norm_scaler = StandardScaler(with_mean=True, with_std=True)
    
    # X_tr = norm_scaler.fit_transform(X_tr)

    # Feature Selection
    # _k_features = 200
    
    # selector = SelectKBest(score_func=f_classif,
    #                        k=_k_features)
    
    # # Classifier
    # _C = 0.1
    # _kernel = 'rbf'
    # _gamma = 'scale'
    
    # classifier = SVC(random_state=42, 
    #                   C=_C, 
    #                   kernel=_kernel, 
    #                   gamma=_gamma)
    
    # # Selection + Classifier
    # model = make_pipeline(selector, classifier)
    
    # # Cross-Validation
    # skf = StratifiedKFold(n_splits=5)
    
    # scores = cross_validate(model, X_tr, y_tr, cv=skf,
    #                         scoring=('accuracy', 'balanced_accuracy', 'roc_auc'),
    #                         return_train_score=True)
    