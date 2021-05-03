from MachineLearning import grid_search_svm_anova, svm_anova_estimators, mlp_anova, mlp_pca
from ScoringMetrics import cv_results, model_best_fts
from DataAssessment import count_best_fts_types
from DataPreparation import make_features_array, add_labels_to_data_array, dataset_split, get_filenames_labels
from sklearn.model_selection import train_test_split
from DataPreparation import get_saved_features
from FeatureSelection import overall_best_fts
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from FeatureSelection import overall_best_fts
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

modes = ['Diagnosis', 'DiagnosisYoung', 'DiagnosisOld', 'DiagnosisMale', 'DiagnosisFemale']
SCORING = 'roc_auc'
montage = 'Bipolar'

aucs_df = pd.DataFrame()
for MODE in modes:
    
    # Initialize figure
    plt.figure()
    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(30,10))

    # Make array
    bdp_ms, conn_ms, gr_ms, asy_ms = get_saved_features(bdp=True, rawConn=False, conn=True, graphs=True, asy=True, montage=montage)
    
    labels, filenames = get_filenames_labels(mode=MODE)
    
    # Make array
    data = make_features_array(filenames, bdp_ms, conn_ms, gr_ms, asy_ms)
    fts_names = data.columns
    labels_names = add_labels_to_data_array(data, labels, mode=MODE)
    dataset = dataset_split(data)
    dataset['MODE'] = MODE
    dataset['SCORING'] = SCORING
    
    X_train = dataset['X_tr']
    y_train = dataset['y_tr']
    
    # Feature Normalization
    norm_scaler = StandardScaler(with_mean=True, with_std=True)
    
    # SVC Model
    svc = SVC(C=0.1, gamma=0.01, kernel = 'rbf', random_state=42, probability=True)
    pipe = Pipeline(steps=[('norm_scaler', norm_scaler), ('classifier', svc)])
    
    skf = StratifiedKFold(n_splits = 5)
    
    aucs = []
    
    for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
    
        X_tr, y_tr = X_train[train_index], y_train[train_index]
        X_val, y_val = X_train[test_index], y_train[test_index]
        
        selector, X_tr_presel, X_val_presel = overall_best_fts(X_tr, y_tr, X_val, fts_names, estimator=pipe, mode=MODE,
                                                        k_features_bdp=20, k_features_graph=150,
                                                        k_features_asy=50, k_features_conn=50,
                                                        n_features_to_select=25, scoring=SCORING, cv=3)
        
        X_tr_sel = selector.transform(X_tr_presel)
        X_val_sel = selector.transform(X_val_presel) 
        
        space = dict({
        'classifier__C': [0.01, 0.1, 0.5, 1, 2, 10],
        'classifier__gamma': [0.01, 0.05, 0.1, 0.5, 1],
        'classifier__kernel': ['rbf', 'sigmoid']
        })
        
        clf = GridSearchCV( estimator=pipe,
                            param_grid=space,
                            scoring=SCORING, 
                            n_jobs=-1,
                            cv=5)
        
        clf.fit(X_tr_sel, y_tr)
        
        print('\nHYPERPARAMETERS')
        print(clf.best_params_, '\n')
        print('TRAINING BEST SCORE')
        print(clf.best_score_, '\n')
    
        estimator_final = clf.best_estimator_
        print('VALIDATION SCORE')
        print(estimator_final.score(X_val_sel, y_val), '\n')
        
        # Probabilities
        y_prob = estimator_final.predict_proba(X_val_sel)[:,1]
        
        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_val, y_prob)
        
        # AUC ROC
        _auc = auc(fpr, tpr)
        
        if _auc < 0.5:
            y_prob = estimator_final.predict_proba(X_val_sel)[:,0]
            fpr, tpr, thresholds = roc_curve(y_val, y_prob)
            _auc = auc(fpr, tpr)
        
        aucs.append(_auc)
        
        # Optimal cut-off point
        max_idxs = np.argmax(tpr - fpr)
        opti_fpr, opti_tpr, opti_th = fpr[max_idxs], tpr[max_idxs], thresholds[max_idxs]
        
        # Display ROC curve
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=aucs[i], estimator_name="SVM+Hybrid")
        display.plot(axs[1,i])
        axs[1,i].scatter(opti_fpr, opti_tpr, s=50, c='orange', alpha=1)
        axs[1,i].plot([0,1], [0,1], '--k')
    
        # Confusion Matrix for optimal threshold
        y_pred = (y_prob > opti_th).astype('float')
        # y_pred = e.predict(X_val)
        confusionMatrix = confusion_matrix(y_val, y_pred)
    
        sb.heatmap(confusionMatrix, annot=True, cmap='Blues', fmt='g', ax=axs[0,i])
        axs[0,i].title.set_text('Threshold: {:.2f}'.format(opti_th))
        axs[0,i].set_xlabel('Target Class')
        axs[0,i].set_ylabel('Predicted Class')
        plt.suptitle(MODE + ' 5-Fold CV ROC Curves & Confusion Matrices (AUC = {:.3f} ± {:.3f})'.format(np.mean(aucs), np.std(aucs)), va='center', fontsize=30)
        
    aucs_df = pd.concat([aucs_df, pd.DataFrame([[MODE]*5, [montage]*5, aucs], index=['Classification', 'Montage', 'AUC']).transpose()], axis=0)

#%%
import seaborn as sb
from matplotlib import pyplot as plt

plt.figure(figsize=(14,7))
box_plot = sb.boxplot(x="Classification", y="AUC", hue='Montage', data=aucs_df, palette=sb.color_palette("hls", 2))

