import numpy as np
from Pickle import getPickleFile
from matplotlib import pyplot as plt
from FeatureSelection import eliminate_corr_fts
from MachineLearning import svm_anova

global MODE, SCORING
MODE = 'Diagnosis'
SCORING = 'roc_auc'

#%% Eliminate highly correlated features - what's the best threshold?

labels_names = getPickleFile('../3_ML_Data/128Hz/labelsNames')

scores = []
ths = np.arange(0.75,1.0,0.01)
for _th in ths:
    dataset = getPickleFile('../3_ML_Data/128Hz/dataset')
    fts_names = getPickleFile('../3_ML_Data/128Hz/featuresNames')
    dataset, fts_names = eliminate_corr_fts(dataset, fts_names, th=_th)
    clf_svm_anova = svm_anova(dataset, labels_names, MODE, SCORING)
    scores.append(clf_svm_anova.best_score_)

plt.figure()
plt.plot(ths, scores)
plt.title('AUC ROC scores vs. threshold in correlated features elimination')
plt.xlabel('Threshold')
plt.ylabel('Score')