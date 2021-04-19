import pandas as pd
from Pickle import getPickleFile, createPickleFile
from PreProcessing import  get_ica_template, eeg_preprocessing, clean_epochs
from EpochSelection import epochs_selection_bandpower
from FeatureExtraction import extract_bandpowers, extract_features, compute_connectivity_measures
from GraphMeasures import compute_graph_subgroup_measures
from Asymmetry import compute_asymmetry_measures
from DataPreparation import get_saved_features,  make_features_array, \
                            add_labels_to_data_array, dataset_split, get_filenames_labels
                            
from PlotDistribution import plot_data_distribution
from PlotTSNE import plot_tsne
from BestRankedFeatures import best_ranked_features
from MachineLearning import svm_anova, svm_pca, mlp_anova, \
                            mlp_pca, rfc_anova, rfc_pca


'''
    New bandpower extractor, now with epoch selection
'''

global filenames
filenames = pd.read_excel('Metadata_train.xlsx')['Filename']

#%% EEG Pre-Processing 256Hz

icas = get_ica_template(filenames[0])

for filename in filenames:
    epochs = eeg_preprocessing(filename, icas, plot=False)
    epochs, _ = clean_epochs(filename, epochs, plot=False)
    createPickleFile(epochs, '../1_PreProcessed_Data/' + filename)

#%% Epochs Downsample to 128Hz

for filename in filenames:
    epochs = getPickleFile('../1_PreProcessed_Data/' + filename)
    epochs.resample(sfreq=128)
    createPickleFile(epochs, '../1_PreProcessed_Data/128Hz/' + filename)

#%% Extraction of Bandpower and Connectivity Features 

BDP = {}
IMCOH = {}
PLV = {}
MI = {}
PDC = {}

# over all subjects
for i, filename in enumerate(filenames):
    saved_epochs = getPickleFile('../1_PreProcessed_Data/128Hz/' + filename)
            
    bd_names, s_epochs = epochs_selection_bandpower(saved_epochs)
    
    BDP[filename] = extract_bandpowers(s_epochs, filename)
    
    # IMCOH[filename], PLV[filename], MI[filename],\
    # PDC[filename] = extract_features(bd_names, s_epochs)
    
    # save features in pickle
    createPickleFile(BDP, '../2_Features_Data/128Hz/' + 'bdp')
    # createPickleFile(IMCOH, '../2_Features_Data/128Hz/' + 'imcoh')
    # createPickleFile(PLV, '../2_Features_Data/128Hz/' + 'plv')
    # createPickleFile(MI, '../2_Features_Data/128Hz/' + 'mi')
    # createPickleFile(PDC, '../2_Features_Data/128Hz/' + 'pdc')         

#%% From connectivity matrices, compute subgroups' measures

#Subgroups Connectivity Features

fts = get_saved_features(bdp=False, rawConn=True, conn=False, graphs=False, asy=False)
conn_ms = compute_connectivity_measures(fts)
createPickleFile(conn_ms, '../2_Features_Data/128Hz/' + 'connectivityMeasures')

# Subgroups Graph Measures

fts = get_saved_features(bdp=False, rawConn=True, conn=False, graphs=False, asy=False)
graph_ms = compute_graph_subgroup_measures(fts)
createPickleFile(graph_ms, '../2_Features_Data/128Hz/' + 'graphMeasures')

# Subgroups Graph Asymmetry Ratios

fts = get_saved_features(bdp=False, rawConn=False, conn=False, graphs=True, asy=False)
asymmetry_ms = compute_asymmetry_measures(fts)
createPickleFile(asymmetry_ms, '../2_Features_Data/128Hz/' + 'asymmetryMeasures')

#%% Working Mode & Generate All Features Matrix
''' 
Diagnosis:                  roc_auc
Epilepsy types:             balanced_accuracy
Gender:                     f1
Age:                        f1
Sleep:                      f1
Diagnosis-Sleep:            f1
CardiovascularVSEpileptic:  f1
DiagnosisWithAgeGender      roc_auc
'''

global MODE, SCORING
MODE = 'Diagnosis'
SCORING = 'roc_auc'

bdp_ms, conn_ms, gr_ms, asy_ms = get_saved_features(bdp=True, rawConn=False, conn=True, graphs=True, asy=True)

labels, filenames = get_filenames_labels(mode=MODE)

# Make array
data = make_features_array(filenames, bdp_ms, conn_ms, gr_ms, asy_ms)
fts_names = data.columns

createPickleFile(data, '../2_Features_Data/128Hz/' + 'allFeatures')
createPickleFile(fts_names, '../3_ML_Data/128Hz/' + 'featuresNames')

labels_names = add_labels_to_data_array(data, labels, mode=MODE)
dataset = dataset_split(data)

createPickleFile(dataset, '../3_ML_Data/128Hz/' + 'dataset')
createPickleFile(labels_names, '../3_ML_Data/128Hz/' + 'labelsNames')
    
#%% TRAIN Machine Learning - get data from Pickle
global dataset, fts_names, labels_names
dataset = getPickleFile('../3_ML_Data/128Hz/dataset')
fts_names = getPickleFile('../3_ML_Data/128Hz/featuresNames')
labels_names = getPickleFile('../3_ML_Data/128Hz/labelsNames')

#%% Plot Data Distribution
c = plot_data_distribution(dataset, labels_names, MODE)

#%% Plot TSNE
# %config InlineBackend.figure_format='retina'
fig_tsne = plot_tsne(dataset, labels_names, MODE)
    
#%% Best Ranked Features
best_fts = best_ranked_features(dataset,fts_names, k_features=50)

#%% GridSearchCV of Best Models (run current line with F9)
# SVM + SelectKBest
clf_svm_anova = svm_anova(dataset, MODE, SCORING)

# SVM + PCA
clf_svm_pca = svm_pca(dataset, MODE, SCORING)

# MLP + SelectKBest
clf_mlp_anova = mlp_anova(dataset, MODE, SCORING)

# MLP + PCA
clf_mlp_pca = mlp_pca(dataset, MODE, SCORING)

# RFC + SelectKBest
clf_rfc_anova = rfc_anova(dataset, MODE, SCORING)

# RFC + PCA
clf_rfc_pca = rfc_pca(dataset, MODE, SCORING)

#%% Model Exhaustive assesment and report


