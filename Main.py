import pandas as pd
from Pickle import getPickleFile, createPickleFile
from DataPreparation import get_saved_features

'''
    Focal Symptomatic RULES
'''

filenames = pd.read_excel('Metadata_train.xlsx')['Filename']

#%% EEG Pre-Processing 256Hz 5s
from PreProcessing import get_ica_template, eeg_preprocessing, clean_epochs

icas = get_ica_template(filenames[0])

for filename in filenames:
    epochs = eeg_preprocessing(filename, icas, epoch_length=5, plot=False)
    epochs, _ = clean_epochs(filename, epochs, plot=False)
    createPickleFile(epochs, '../1_PreProcessed_Data/Monopolar/5s/256Hz/' + filename)

#%% EEG Bipolar montage
from PreProcessing import set_bipolar

epochs_lengths = ['2.5s', '5s']

for ep_l in epochs_lengths:
    for filename in filenames[[152]]:
        epochs = getPickleFile('../1_PreProcessed_Data/Monopolar/' + ep_l + '/256Hz/' + filename)
        epochs_b = set_bipolar(epochs)
        createPickleFile(epochs_b, '../1_PreProcessed_Data/Bipolar/' + ep_l + '/256Hz/' + filename)

#%% Extraction of Bandpower and Connectivity Features 
from EpochSelection import epochs_selection_bandpower
from PreProcessing import resample_epochs
from FeatureExtraction import extract_bandpowers, extract_features

# BDP, IMCOH, PLV, MI, PDC = {}, {}, {}, {}, {}

BDP = getPickleFile('../2_Features_Data/Bipolar/' + 'bdp_256')
IMCOH = getPickleFile('../2_Features_Data/Bipolar/' + 'imcoh')
PLV = getPickleFile('../2_Features_Data/Bipolar/' + 'plv')
MI = getPickleFile('../2_Features_Data/Bipolar/' + 'mi')
PDC = getPickleFile('../2_Features_Data/Bipolar/' + 'pdc') 

# over all subjects
for i, filename in enumerate(filenames):
    # bandpower extraction on 5s epochs, 256Hz
    saved_epochs = getPickleFile('../1_PreProcessed_Data/Bipolar/5s/256Hz/' + filename)
    _, s_epochs = epochs_selection_bandpower(saved_epochs)
    
    BDP[filename] = extract_bandpowers(s_epochs, filename)
    
    # functional connectivity on 2.5s epochs, 128Hz
    saved_epochs = getPickleFile('../1_PreProcessed_Data/Bipolar/2.5s/256Hz/' + filename)
    downsampled_epochs = resample_epochs(epochs, sfreq=128)
    bd_names, s_epochs = epochs_selection_bandpower(downsampled_epochs)
    
    IMCOH[filename], PLV[filename], MI[filename],\
    PDC[filename] = extract_features(bd_names, s_epochs)
    
    # save features in pickle
    createPickleFile(BDP, '../2_Features_Data/Bipolar/' + 'bdp_256')
    createPickleFile(IMCOH, '../2_Features_Data/Bipolar/' + 'imcoh')
    createPickleFile(PLV, '../2_Features_Data/Bipolar/' + 'plv')
    createPickleFile(MI, '../2_Features_Data/Bipolar/' + 'mi')
    createPickleFile(PDC, '../2_Features_Data/Bipolar/' + 'pdc')  
  
#%% From connectivity matrices, compute subgroups' measures
#Subgroups Connectivity Features
from FeatureExtraction import compute_connectivity_measures
fts = get_saved_features(bdp=False, rawConn=True, conn=False, graphs=False, asy=False)
conn_ms = compute_connectivity_measures(fts)
createPickleFile(conn_ms, '../2_Features_Data/Bipolar/' + 'connectivityMeasures')

#% Subgroups Graph Measures
from GraphMeasures import compute_graph_subgroup_measures
fts = get_saved_features(bdp=False, rawConn=True, conn=False, graphs=False, asy=False)
graph_ms = compute_graph_subgroup_measures(fts)
createPickleFile(graph_ms, '../2_Features_Data/Bipolar/' + 'graphMeasures')

#%% Subgroups Graph Asymmetry Ratios
from Asymmetry import compute_asymmetry_measures
fts = get_saved_features(bdp=False, rawConn=False, conn=False, graphs=True, asy=False)
asymmetry_ms = compute_asymmetry_measures(fts)
createPickleFile(asymmetry_ms, '../2_Features_Data/Bipolar/' + 'asymmetryMeasures')

#%% Working Mode & Generate All Features Matrix
''' 
Diagnosis:                      roc_auc

Epilepsy types:                 balanced_accuracy
Gender:                         f1
Age:                            f1
Sleep:                          f1
Diagnosis-Sleep:                f1

CardiovascularVSEpileptic:      f1
ProvokedVSEpileptic             roc_auc
PsychogenicVSEpileptic          roc_auc
VagalSyncopeVSEpileptic         roc_auc

DiagnosisMale                   roc_auc
DiagnosisFemale                 roc_auc
DiagnosisYoung                  roc_auc
DiagnosisOld                    roc_auc

AntecedentFamilyEpileptic       
AntecedentFamilyNonEpileptic    
AntecedentFamilyOther

AntecedentChildDevelopDisorder  
AntecedentChildFebrileSeizure   
AntecedentChildMyoclonus        
AntecedentChildNone             
AntecedentChildOther  

'FocalSymptomaticVSNon-Epileptic'
'FocalSymptomaticVSNon-EpilepticYoung'
'FocalSymptomaticVSNon-EpilepticOld'
'FocalSymptomaticVSNon-EpilepticMale'
'FocalSymptomaticVSNon-EpilepticFemale'
          
'''

global MODE, SCORING
MODE = 'FocalSymptomaticVSNon-Epileptic'
SCORING = 'roc_auc'

#%% Make features array
from DataPreparation import make_features_array, add_labels_to_data_array, dataset_split, get_filenames_labels
bdp_ms, conn_ms, gr_ms, asy_ms = get_saved_features(bdp=True, rawConn=False, conn=True, graphs=True, asy=True)

labels, filenames = get_filenames_labels(mode=MODE)

# Make array
data = make_features_array(filenames, bdp_ms, conn_ms, gr_ms, asy_ms)
fts_names = data.columns

createPickleFile(data, '../2_Features_Data/Bipolar/' + 'allFeatures')
createPickleFile(fts_names, '../3_ML_Data/Bipolar/' + 'featuresNames')

labels_names = add_labels_to_data_array(data, labels, mode=MODE)
dataset = dataset_split(data)
dataset['MODE'] = MODE
dataset['SCORING'] = SCORING

createPickleFile(dataset, '../3_ML_Data/Bipolar/' + 'dataset')
createPickleFile(labels_names, '../3_ML_Data/Bipolar/' + 'labelsNames')

#%% Eliminate highly correlated features
from FeatureSelection import eliminate_corr_fts   
dataset, fts_names = eliminate_corr_fts(dataset, fts_names, th=0.92)

#%% TRAIN Machine Learning - get data from Pickle
dataset = getPickleFile('../3_ML_Data/Bipolar/dataset')
fts_names = getPickleFile('../3_ML_Data/Bipolar/featuresNames')
labels_names = getPickleFile('../3_ML_Data/Bipolar/labelsNames')
MODE = dataset['MODE']
SCORING = dataset['SCORING']

#%% Preliminary Data Assessment and Predictive Power
from DataAssessment import plot_data_distribution, plot_tsne, best_ranked_features, fts_correlation_matrix, most_least_correlated_fts
                        
# Plot Data Distribution for Current MODE
fig_data_dist = plot_data_distribution(dataset, labels_names, MODE)

# Plot TSNE
#%config InlineBackend.figure_format='retina'
fig_tsne = plot_tsne(dataset, labels_names, MODE, p=30)
    
# Best Ranked Features
best_fts = best_ranked_features(dataset,fts_names, k_features=100)
    
# Features Correlation Matrix
corr_df = fts_correlation_matrix(dataset, fts_names, ms_keep=[], ms_exclude=[], k_best_features=9)

# Most and Least Correlated Features
corr_most, corr_least = most_least_correlated_fts(dataset, fts_names, n=-1)

#%% Train Undersampling
from imblearn.under_sampling import RandomUnderSampler 
bl = RandomUnderSampler(random_state=42)
dataset['X_tr'], dataset['y_tr'] = bl.fit_resample(dataset['X_tr'], dataset['y_tr'])  
fig_data_dist = plot_data_distribution(dataset, labels_names, MODE)   

#%% Machine Learning (run current line with F9)
from ScoringMetrics import cv_results, model_best_fts
from DataAssessment import count_best_fts_types

#%% SVM & SelectKBest
from MachineLearning import grid_search_svm_anova, svm_anova_estimators

gs_svm_anova, model, clf = grid_search_svm_anova(dataset)
estimators_svm_anova = svm_anova_estimators(dataset, gs_svm_anova, model)
aucs = cv_results(dataset, estimators_svm_anova, model)
best_features = model_best_fts(dataset, fts_names, estimators_svm_anova)
count_best_fts_types(best_features, MODE)

#%% SVM & PCA
from MachineLearning import grid_search_svm_pca, svm_pca_estimators

gs_svm_pca, model, clf = grid_search_svm_pca(dataset, labels_names)
estimators_svm_pca = svm_pca_estimators(dataset, gs_svm_pca, model)
aucs = cv_results(dataset, estimators_svm_pca, model)

#%% MLP & SelectKBest
from MachineLearning import grid_search_mlp_anova, mlp_anova_estimators

gs_mlp_anova, model, clf = grid_search_mlp_anova(dataset, labels_names)
estimators_mlp_anova = mlp_anova_estimators(dataset, gs_mlp_anova, model)
aucs = cv_results(dataset, estimators_mlp_anova, model)
best_features = model_best_fts(dataset, fts_names, estimators_mlp_anova)
count_best_fts_types(best_features, MODE)

#%% MLP & PCA
from MachineLearning import grid_search_mlp_pca, mlp_pca_estimators

gs_mlp_pca, model, clf = grid_search_mlp_pca(dataset, labels_names)
estimators_mlp_pca = mlp_pca_estimators(dataset, gs_mlp_pca, model)
aucs = cv_results(dataset, estimators_mlp_pca, model)

#%% RFC & SelectKBest
from MachineLearning import grid_search_rfc_anova, rfc_anova_estimators

gs_rfc_anova, model, clf = grid_search_rfc_anova(dataset, labels_names)
estimators_rfc_anova = rfc_anova_estimators(dataset, gs_rfc_anova, model)
aucs = cv_results(dataset, estimators_rfc_anova, model)
best_features = model_best_fts(dataset, fts_names, estimators_rfc_anova)
count_best_fts_types(best_features, MODE)

#%% RFC & PCA
from MachineLearning import grid_search_rfc_pca, rfc_pca_estimators

gs_rfc_pca, model, clf = grid_search_rfc_pca(dataset, labels_names)
estimators_rfc_pca = rfc_pca_estimators(dataset, gs_rfc_pca, model)
aucs = cv_results(dataset, estimators_rfc_pca, model)

#%% LogReg & SelectKBest
from MachineLearning import grid_search_logReg_anova, logReg_anova_estimators

gs_logReg_anova, model, clf = grid_search_logReg_anova(dataset)
estimators_logReg_anova = logReg_anova_estimators(dataset, gs_logReg_anova, model)
aucs = cv_results(dataset, estimators_logReg_anova, model)
best_features = model_best_fts(dataset, fts_names, estimators_logReg_anova)
count_best_fts_types(best_features, MODE)

#%% LogReg & PCA
from MachineLearning import grid_search_logReg_pca, logReg_pca_estimators

gs_logReg_pca, model, clf = grid_search_logReg_pca(dataset, labels_names)
estimators_logReg_pca = logReg_pca_estimators(dataset, gs_logReg_pca, model)
aucs = cv_results(dataset, estimators_logReg_pca, model)

#%% Compare models and boxplot 
from ScoringMetrics import compare_models_montages, boxplot_models
modes = ['FocalSymptomaticVSNon-Epileptic', 
         'FocalSymptomaticVSNon-EpilepticYoung', 
         'FocalSymptomaticVSNon-EpilepticOld', 
         'FocalSymptomaticVSNon-EpilepticMale', 
         'FocalSymptomaticVSNon-EpilepticFemale']

montages = ['Bipolar']
SCORING = 'roc_auc'

aucs_df = compare_models_montages(dataset, modes, montages)
boxplot_models(aucs_df)

#%% Optimize correlation-based elimination threshold
from ScoringMetrics import optimize_eliminate_corr
from MachineLearning import grid_search_logReg_anova
from MachineLearning import grid_search_svm_anova

ths, tr_scores, val_scores, opti_th, best_score = optimize_eliminate_corr(grid_search_svm_anova)


