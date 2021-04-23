from sklearn.metrics import confusion_matrix, plot_roc_curve, classification_report
 
import seaborn as sb
import matplotlib.pyplot as plt


def plot_confusion_matrix(dataset, clf, mode='Diagnosis', model='SVM + ANOVA', scoring='roc_auc'):
    
    y_true = dataset['y_tr']
    y_pred = clf.predict(dataset['X_tr'])
    
    confusionMatrix = confusion_matrix(y_true, y_pred)
    
    plt.figure()
    sb.heatmap(confusionMatrix, annot=True, cmap='Blues', fmt='g')
    plt.title(mode + ' ' + model + ' ' + scoring)
    plt.xlabel('Target Class')
    plt.ylabel('Predicted Class')


def plot_roc(dataset, clf, mode='Diagnosis', model='SVM + ANOVA', scoring='roc_auc'):
    
    X_tr = dataset['X_tr']
    y_tr = dataset['y_tr']
    
    plt.figure()
    plot_roc_curve(estimator=clf, X=X_tr, y=y_tr)
    plt.title(mode + ' ' + model + ' ' + scoring)

def print_metric_report(dataset, labels_names, clf):
    
    y_true = dataset['y_tr']
    y_pred = clf.predict(dataset['X_tr'])
    
    rep = classification_report(y_true, y_pred, target_names=labels_names, zero_division=0)
    
    print(rep)
    
def assess_model(dataset, clf, labels_names, mode='Diagnosis', model='SVM + ANOVA', scoring='roc_auc'):
    print('\n' + mode + ' ' + model)
    print('\nValidation Score with ' + scoring + ':')
    print('{:.3f}'.format(clf.best_score_))
    print('\nModel Parameters:')
    print(clf.best_params_, '\n')
    # print_metric_report(dataset, labels_names, clf)
    
    # plot_confusion_matrix(dataset, clf, mode=mode, model=model, scoring=scoring)
    # plot_roc(dataset, clf, mode=mode, model=model, scoring=scoring)
    
