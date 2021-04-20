from DataPreparation import get_saved_features, make_features_array, add_labels_to_data_array, dataset_split, get_filenames_labels
from DataAssessment import plot_data_distribution, plot_tsne, best_ranked_features, fts_correlation_matrix, most_least_correlated_fts
from FeatureSelection import eliminate_corr_fts   
from MachineLearning import svm_anova, svm_pca, mlp_anova, mlp_pca

modes = ['DiagnosisMale', 'DiagnosisFemale', 'DiagnosisYoung', 'DiagnosisOld']

SCORING = 'roc_auc'

best_ftss= []
corrs = []

for MODE in modes:
    
    bdp_ms, conn_ms, gr_ms, asy_ms = get_saved_features(bdp=True, rawConn=False, conn=True, graphs=True, asy=True)
    
    labels, filenames = get_filenames_labels(mode=MODE)

    data = make_features_array(filenames, bdp_ms, conn_ms, gr_ms, asy_ms)
    fts_names = data.columns
    
    labels_names = add_labels_to_data_array(data, labels, mode=MODE)
    dataset = dataset_split(data)
    
    fig_data_dist = plot_data_distribution(dataset, labels_names, MODE)
    fig_tsne = plot_tsne(dataset, labels_names, MODE)
    best_ftss.append(best_ranked_features(dataset,fts_names, k_features=100))
    corrs.append(most_least_correlated_fts(dataset, fts_names, n=100, ms_keep=[], ms_exclude=[]))

    dataset, fts_names = eliminate_corr_fts(dataset, fts_names, th=1)
    
    clf_svm_anova = svm_anova(dataset, labels_names, MODE, SCORING)
    clf_svm_pca = svm_pca(dataset, labels_names, MODE, SCORING)
    clf_mlp_anova = mlp_anova(dataset, labels_names, MODE, SCORING)
    clf_mlp_pca = mlp_pca(dataset, labels_names, MODE, SCORING)
    
    