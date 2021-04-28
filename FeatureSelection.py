import pandas as pd
import numpy as np
from Pickle import getPickleFile
from DataAssessment import fts_correlation_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from DataAssessment import best_ranked_features
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

#%%

def eliminate_corr_fts(dataset, fts_names, th=0.95):

    X_tr = dataset['X_tr']
    data = pd.DataFrame(data=X_tr, columns=fts_names)
    
    # feature normalization for correlation matrix
    norm_scaler = StandardScaler(with_mean=True, with_std=True)
    minMax_scaler = MinMaxScaler(feature_range=(0,1))
    
    X_tr_norm = norm_scaler.fit_transform(X_tr)
    X_tr_norm = minMax_scaler.fit_transform(X_tr_norm)
    
    corr = pd.DataFrame(data=X_tr_norm, columns=fts_names).corr() 
    
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if corr.iloc[i,j] >= th:
                if columns[j]:
                    columns[j] = False
                    
    fts_names_new = data.columns[columns]
    dataset['X_tr'] = data[fts_names_new].to_numpy()
        
    # do the same for the test set
    X_ts = dataset['X_ts'] 
    data = pd.DataFrame(data=X_ts, columns=fts_names)
    dataset['X_ts'] = data[fts_names_new].to_numpy()
    
    
    return dataset, fts_names_new

#%% ANOVA for Feature Pre-Selection + Sequential Feature Selection
def overall_best_fts(X_tr, y_tr, X_val, fts_names, estimator, k_features_bdp=20,
                     k_features_graph=150, k_features_asy=50,
                     k_features_conn=50, n_features_to_select=50,
                     scoring='roc_auc', cv=5):
    
    fts_types = []
    fts_names_bdp = []
    fts_names_graph = []
    fts_names_asy = []
    fts_names_conn = []
    
    
    fts_type_list_conn = ['imcoh', 'plv', 'mi', 'pdc']
    fts_type_list_graph = ['betweness_centr', 'clustering_coef', 'incoming_flow',
                          'outgoing_flow', 'node_strengths', 'efficiency']
    for fts in fts_names:
        fts_split = (fts.split('-'))
        conn=[i for i in fts_type_list_conn if i == fts_split[0]]
        graph=[i for i in fts_type_list_graph if i == fts_split[0] and 'vs' not in fts_split[3]]
        asymmetry=[i for i in fts_type_list_graph if i == fts_split[0] and 'vs' in fts_split[3]]
        if conn!=[]:
            fts_types.append('Connectivity')
        elif fts_split[0]=='bdp':
            fts_types.append('Bandpowers')
        elif graph!=[]:
            fts_types.append('GraphMeasures')
        elif asymmetry!=[]:
            fts_types.append('Asymmetry')
    
    a=0
    b=0
    c=0
    d=0
    
    for i, fts_type in enumerate(fts_types):
        if fts_type == 'Bandpowers':
            if a==0:
                X_tr_bdp = X_tr[:, i].reshape(-1,1)
                a=a+1
            else:
                X_tr_bdp = np.append(X_tr_bdp, X_tr[:, i].reshape(-1,1), axis=1)
            fts_names_bdp.append(fts_names[i])
                
        elif fts_type == 'GraphMeasures':
            if b==0:
                X_tr_graph = X_tr[:, i].reshape(-1,1)
                b=b+1
            else:
                X_tr_graph = np.append(X_tr_graph, X_tr[:, i].reshape(-1,1), axis=1)
            fts_names_graph.append(fts_names[i])
                
        elif fts_type == 'Asymmetry':
            if c==0:
                X_tr_asy = X_tr[:, i].reshape(-1,1)
                c=c+1
            else:
                X_tr_asy = np.append(X_tr_asy, X_tr[:, i].reshape(-1,1), axis=1)
            fts_names_asy.append(fts_names[i])
        elif fts_type == 'Connectivity':
            if d==0:
                X_tr_conn = X_tr[:, i].reshape(-1,1)
                d=d+1
            else:
                X_tr_conn = np.append(X_tr_conn, X_tr[:, i].reshape(-1,1), axis=1)
            fts_names_conn.append(fts_names[i])
                
    dataset_bdp = {'X_tr': X_tr_bdp, 'y_tr': y_tr}
    dataset_graph = {'X_tr': X_tr_graph, 'y_tr': y_tr}
    dataset_asy = {'X_tr': X_tr_asy, 'y_tr': y_tr}
    dataset_conn = {'X_tr': X_tr_conn, 'y_tr': y_tr}
    fts_names_bdp = pd.Index(fts_names_bdp)
    fts_names_graph = pd.Index(fts_names_graph)
    fts_names_asy = pd.Index(fts_names_asy)
    fts_names_conn = pd.Index(fts_names_conn)
        
    if k_features_bdp>np.shape(X_tr_bdp)[1]:
        k_features_bdp = np.shape(X_tr_bdp)[1]
    best_ranked_features_bdp = best_ranked_features(dataset_bdp, fts_names_bdp, k_features_bdp)
    
    if k_features_graph>np.shape(X_tr_graph)[1]:
        k_features_graph = np.shape(X_tr_graph)[1]
    best_ranked_features_graph = best_ranked_features(dataset_graph, fts_names_graph, k_features_graph)
    
    if k_features_asy>np.shape(X_tr_asy)[1]:
        k_features_asy = np.shape(X_tr_asy)[1]
    best_ranked_features_asy = best_ranked_features(dataset_asy, fts_names_asy, k_features_asy)
    
    if k_features_conn>np.shape(X_tr_conn)[1]:
        k_features_conn = np.shape(X_tr_conn)[1]
    best_ranked_features_conn = best_ranked_features(dataset_conn, fts_names_conn, k_features_conn)
    
    
    idx_brf = []
    for i, fts_name in enumerate(fts_names):
        if fts_name in best_ranked_features_bdp['fts_names'].tolist() or fts_name in best_ranked_features_graph['fts_names'].tolist() or fts_name in best_ranked_features_asy['fts_names'].tolist() or fts_name in best_ranked_features_conn['fts_names'].tolist():
            idx_brf.append(i)
        
    fts_names_pre_selected= fts_names[idx_brf]
    X_tr_pre_selected=X_tr[:, idx_brf]
    X_val_pre_selected=X_val[:,idx_brf]
    
    selector = SequentialFeatureSelector(estimator=estimator,\
                                         n_features_to_select=n_features_to_select,\
                                         scoring=scoring, cv=cv, n_jobs=-1)
    X_tr_pre_selected = selector.fit_transform(X_tr_pre_selected, y_tr)
    
    idx = selector.get_support(indices=True)
    X_val_pre_selected=X_val_pre_selected[:,idx]
    best_fts = pd.DataFrame(data=fts_names_pre_selected[idx], index=idx, columns=['fts_names'])
    
    
    return X_tr_pre_selected, X_val_pre_selected, best_fts


