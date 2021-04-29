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
    
    aucs, y_probs, roc_aucs = [], [], []
    
    for i, (e, (train_index, test_index)) in enumerate(zip(estimators, skf.split(X_tr, y_tr))):
        X_val, y_val = X_tr[test_index], y_tr[test_index]
        
        # Probabilities
        y_prob = e.predict_proba(X_val)[:,1]
        
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

def model_best_fts(dataset, fts_names, estimators):
    
    allBestFts = pd.DataFrame()
    
    for e in estimators:
        selector = e.steps[1][1]
        allBestFts = pd.concat([allBestFts, _best_fts(selector, fts_names)], axis=0)
        
    return allBestFts.sort_values(by='score', ascending=False)