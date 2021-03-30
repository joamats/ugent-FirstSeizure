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
from sklearn.decomposition import PCA
import random
from datetime import datetime
# import seaborn as sb
# import pandas as pd

datasets = getPickleFile('../ML_Data/' + 'datasets_noStd')
# scores_MLP = []

selector = SelectKBest(score_func=f_classif)

parameters={'selector__k': [50, 70, 90, 100, 300, 700, 1000],
                'classifier__hidden_layer_sizes':[(20,), (50,), (100,), (150,), 
                                                  (20,20),(50,50),(100,100), (150,150),
                                                  (20,20,20),(50,50,50), (100,100,100),
                                                  (150,150,150), (50,100,50)],
                'classifier__activation': ['relu'],
                'classifier__solver': ['adam'],
                'classifier__learning_rate': ["constant", "invscaling", "adaptive"],
                'classifier__alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]}

score=[]
random_feat=[[-1,-1,-1,-1]]
for i in range(50):
    random.seed(datetime.now())
    k=hls=lr=alpha=-1
    while([k,hls, lr, alpha] in random_feat):
        k=random.randint(0, np.size(parameters['selector__k'])-1)
        hls=random.randint(0, np.size(parameters['classifier__hidden_layer_sizes'])-1)
        lr=random.randint(0, np.size(parameters['classifier__learning_rate'])-1)
        alpha=random.randint(0, np.size(parameters['classifier__alpha'])-1)
    random_feat.append([k,hls, lr, alpha])
    
    recall_MLP = []
    precision_MLP = []
    roc_auc_MLP = []
    
    for dset in datasets:
    
        X_tr = dset['X_tr']
        y_tr = dset['y_tr']
       
        # X_embedded = TSNE(n_components=2).fit_transform(X_tr)
         
        # df = pd.DataFrame()
        # df['one'] = X_embedded[:,0]
        # df['two'] = X_embedded[:,1]
        # df['y'] = y_tr
        
        # sb.scatterplot(
        #     x="one", y="two",
        #     hue="y",
        #     palette=sb.color_palette("hls", 2),
        #     data=df,
        #     legend="full",
        #     alpha=0.8
        # )
       
        #Feature Normalization
        norm_scaler = StandardScaler(with_mean=True, with_std=True)
        minMax_scaler = MinMaxScaler()
    
        #Feature Selection
        # _k_features = 70
        
        # selector = SelectKBest(score_func=f_classif,
        #                         k=_k_features)
        # selector = PCA(random_state=42, n_components = 150)
        # # SVC
        # _C = 1.5
        # _kernel = 'rbf'
        
        # SVC_classifier = SVC(random_state=42, C=_C, kernel=_kernel)
        
        # #RFC
        # _criterion = 'entropy'
        # _n_estimators = 800
        # _bootstrap = False
        # _max_depth = 80
        # _max_features = 'auto'
        # _min_samples_leaf = 1
        # _min_samples_split = 10
        # RFC_classifier = RandomForestClassifier(random_state=42,
        #                                         criterion = _criterion,
        #                                         n_estimators = _n_estimators)
        
        #NN
        parameters_dict = {'selector__k': parameters['selector__k'][k],
                'classifier__hidden_layer_sizes': parameters['classifier__hidden_layer_sizes'][hls],
                'classifier__activation': ['relu'],
                'classifier__solver': ['adam'],
                'classifier__learning_rate': parameters['classifier__learning_rate'][lr],
                'classifier__alpha':parameters['classifier__alpha'][alpha]}
        _activation = 'relu'
        _hidden_layer_sizes = parameters_dict['classifier__hidden_layer_sizes']
        _max_iter = 600
        _alpha = parameters_dict['classifier__alpha']
        _learning_rate = parameters_dict['classifier__learning_rate']
        _solver = 'adam'
        MLP_classifier = MLPClassifier(random_state=42, activation = _activation,
                                        hidden_layer_sizes = _hidden_layer_sizes,
                                        max_iter = _max_iter, alpha = _alpha,
                                        learning_rate = _learning_rate,
                                        solver = _solver)
        
        # Cross-Validation
        skf = StratifiedKFold(n_splits=5)
        
        # models
        # model_SVC = make_pipeline(norm_scaler, minMax_scaler, selector,
        #                           SVC_classifier)
        # model_RFC = make_pipeline(norm_scaler, minMax_scaler, selector,
        #                           RFC_classifier)
        model_MLP = make_pipeline(norm_scaler, minMax_scaler, selector,
                                  MLP_classifier)
        
        # Scores
        # score_SVC = cross_validate(model_SVC, X_tr, y_tr, cv=skf,
        #                         scoring=('accuracy', 'balanced_accuracy', 'roc_auc'),
        #                        return_train_score=True)
        # score_RFC = cross_validate(model_RFC, X_tr, y_tr, cv=skf,
        #                         scoring=('accuracy', 'balanced_accuracy', 'roc_auc'),
        #                         return_train_score=True)
        # scores_RFC.append(score_RFC)
        score_MLP = cross_validate(model_MLP, X_tr, y_tr, cv=skf,
                                scoring=('recall', 'precision', 'roc_auc'),
                                return_train_score=True)
        recall_MLP.append(score_MLP['test_recall'])
        precision_MLP.append(score_MLP['test_precision'])
        roc_auc_MLP.append(score_MLP['test_roc_auc'])
    
    score_dict = {'recall_mean': np.mean(recall_MLP), 'recall_std': np.std(recall_MLP),
                  'precision_mean': np.mean(precision_MLP), 'precision_std': np.std(precision_MLP),
                  'roc_auc_mean': np.mean(roc_auc_MLP), 'roc_auc_std': np.std(roc_auc_MLP),
                  'parameters': parameters_dict}
    score.append(score_dict)

score_list = [x['roc_auc_mean'] for x in score]
print(max(score_list))