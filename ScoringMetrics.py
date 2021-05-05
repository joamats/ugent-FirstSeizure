from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
    
def cv_results(dataset, estimators, model):
    # Dataset
    X_tr = dataset['X_tr']
    y_tr = dataset['y_tr']
    MODE = dataset['MODE']
    
    # Cross-Validation
    skf = StratifiedKFold(n_splits=5)
    
    # Initialize figure
    plt.figure()
    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(30,10))
    
    aucs, accuracies, sensitivities, specificities  = [], [], [], []
    
    for i, (e, (train_index, test_index)) in enumerate(zip(estimators, skf.split(X_tr, y_tr))):
        X_val, y_val = X_tr[test_index], y_tr[test_index]
        
        # Probabilities
        y_prob = e.predict_proba(X_val)[:,1]
        
        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_val, y_prob)
        
        # AUC ROC
        _auc = auc(fpr, tpr)
        
        if _auc < 0.5:
            y_prob = e.predict_proba(X_val)[:,0]
            fpr, tpr, thresholds = roc_curve(y_val, y_prob)
            _auc = auc(fpr, tpr)
        
        aucs.append(_auc)
        
        # Optimal cut-off point
        max_idxs = np.argmax(tpr - fpr)
        opti_fpr, opti_tpr, opti_th = fpr[max_idxs], tpr[max_idxs], thresholds[max_idxs]
        
        # ROC surrogates
        nsurro = 100
        fpr_surros, tpr_surros, auc_surros = [], [], []
        for j in range(nsurro):
            fpr_IEA_su, tpr_IEA_su, _ = roc_curve(y_val, np.random.permutation(y_prob))
            fpr_surros.append(fpr_IEA_su)
            tpr_surros.append(tpr_IEA_su)
            auc_surros.append(auc(fpr_IEA_su, tpr_IEA_su))
            
        # To check for statistical significance, we take percentile 95%
        auc_95 = np.percentile(auc_surros, 95)
        idx_95 = (np.abs(auc_surros - auc_95)).argmin()
        auc_05 = np.percentile(auc_surros, 5)
        idx_05 = (np.abs(auc_surros - auc_05)).argmin()
        
        # Display ROC curve
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=aucs[i], estimator_name=model)
        display.plot(axs[1,i])
        axs[1,i].scatter(opti_fpr, opti_tpr, s=50, c='green', alpha=1, label='Optimal Cut-Off Point')
        axs[1,i].plot([0,1], [0,1], '--k')
        
        # Add Surrogates to plot
        display_surros = RocCurveDisplay(fpr=fpr_surros[idx_95], tpr=tpr_surros[idx_95], roc_auc=auc_95, estimator_name="Surrogates")
        display_surros.plot(axs[1,i])
    
        # Confusion Matrix for optimal threshold
        y_pred = (y_prob > opti_th).astype('float')
        confusionMatrix = confusion_matrix(y_val, y_pred)
        
        TN = confusionMatrix[0][0]
        TP = confusionMatrix[1][1]
        FN = confusionMatrix[0][1]
        FP = confusionMatrix[1][0]
        
        #Accuracy computation
        accuracies.append((TP + TN)/(TP + TN + FP + FN))
        
        #Sensitivity computation
        sensitivities.append(TP / (TP + FN))
        
        #Specificity computation
        specificities.append(TN / (TN + FP))
    
        sb.heatmap(confusionMatrix, annot=True, cmap='Blues', fmt='g', ax=axs[0,i])
        axs[0,i].title.set_text('Optimal Cut-Off Point: {:.2f}'.format(opti_th))
        axs[0,i].set_xlabel('Target Class')
        axs[0,i].set_ylabel('Predicted Class')
        
    plt.suptitle(MODE + ' 5-Fold CV ROC Curves & Confusion Matrices (AUC = {:.3f} ± {:.3f})'.format(np.mean(aucs), np.std(aucs)), va='center', fontsize=30)
    
    print('AUC = {:.3f} ± {:.3f}'.format(np.mean(aucs), np.std(aucs)))
    print('ACCURACY = {:.3f} ± {:.3f}'.format(np.mean(accuracies), np.std(accuracies)))
    print('SENSITIVITY = {:.3f} ± {:.3f}'.format(np.mean(sensitivities), np.std(sensitivities)))
    print('SPECIFICITY = {:.3f} ± {:.3f}'.format(np.mean(specificities), np.std(specificities)))

    return aucs

#%%
def cv_results_hybrid(datasets, estimators, model):
    for i, dataset in enumerate(datasets):
        # Dataset
        X_tr = dataset['X_tr']
        y_tr = dataset['y_tr']
        MODE = dataset['MODE']
    
        # Cross-Validation
        skf = StratifiedKFold(n_splits=5)
        
        # Initialize figure
        plt.figure()
        fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(30,10))
        
        aucs, y_probs, roc_aucs = [], [], []
        
        for i, (train_index, test_index) in enumerate(skf.split(X_tr, y_tr)):
            X_val, y_val = X_tr[test_index], y_tr[test_index]
            
            # Probabilities
            y_prob = estimators[i].predict_proba(X_val)[:,1]
            
            # ROC curve
            fpr, tpr, thresholds = roc_curve(y_val, y_prob)
            
            # Optimal cut-off point
            max_idxs = np.argmax(tpr - fpr)
            opti_fpr, opti_tpr, opti_th = fpr[max_idxs], tpr[max_idxs], thresholds[max_idxs]
            
            # AUC ROC
            aucs.append(auc(fpr, tpr))
            
            # Display ROC curve
            display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=aucs[i], estimator_name=model)
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

    return aucs


#%% Best model's features

from DataAssessment import _best_fts

def model_best_fts(fts_names, estimators, model = 'not_rfc'):
    allBestFts = pd.DataFrame()
    
    if model == 'rfc_builtIn':
        bestFts = pd.DataFrame()
        bestFtsNames = []
        best_features_indexes = []
        
        for e in estimators:
            for ind in e.steps[-1][1].feature_importances_.argsort()[-5:][::-1]:
                best_features_indexes.append(ind)
        for ft_index in best_features_indexes:
            bestFtsNames.append(fts_names[ft_index])
        bestFts['fts_names']=bestFtsNames
        allBestFts = pd.concat([allBestFts, bestFts], axis=0)
        return allBestFts
    
    for e in estimators:
        selector = e.steps[-2][1]
        allBestFts = pd.concat([allBestFts, _best_fts(selector, fts_names)], axis=0)
        
    return allBestFts.sort_values(by='score', ascending=False)

#%% Compare different modes' models

from MachineLearning import grid_search_svm_anova, svm_anova_estimators, grid_search_mlp_anova, mlp_anova_estimators, grid_search_mlp_pca, mlp_pca_estimators, grid_search_rfc_anova, rfc_anova_estimators, grid_search_rfc_pca, rfc_pca_estimators, grid_search_logReg_anova, logReg_anova_estimators, grid_search_logReg_pca, logReg_pca_estimators
from ScoringMetrics import cv_results, model_best_fts
from DataAssessment import count_best_fts_types
from DataPreparation import make_features_array, add_labels_to_data_array, dataset_split, get_filenames_labels

def compare_modes_montages(modes, montages):
    
    for montage in montages:
        for MODE in modes:
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
            
            # ML
            clf_pca = svm_pca(dataset, labels_names)
            gs_svm_anova, model, gs = grid_search_svm_anova(dataset, labels_names)
            estimators_svm_anova = svm_anova_estimators(dataset, gs_svm_anova, model)
            aucs = cv_results(dataset, estimators_svm_anova, model)
            best_features = model_best_fts(dataset, fts_names, estimators_svm_anova)
            count_best_fts_types(best_features, MODE)
            
            aucs_df = pd.concat([aucs_df, pd.DataFrame([[MODE]*5, [montage]*5, aucs], index=['Classification', 'Montage', 'AUC']).transpose()], axis=0)
        
    return aucs_df

#%% Boxplot best models
import seaborn as sb
from matplotlib import pyplot as plt

def boxplot_models(aucs_df):
    plt.figure(figsize=(14,7))
    box_plot = sb.boxplot(x="Classification", y="AUC", hue='Montage', data=aucs_df, palette=sb.color_palette("hls", 2))
    plt.title('Overall results for Focal Symptomatic Epilepsy (SVM & ANOVA)')
    plt.xticks(range(0,5),['All', 'Young', 'Old', 'Male', 'Female'])
    
    
    