import pandas as pd
import numpy as np
import brainconn
from DataPreparation import get_saved_features
from Pickle import createPickleFile
from FeatureExtraction import _features_subgroup_combination

#%% Auxiliary functions

# Compute mean, std, median, ratio for left and right subgroups
def _compute_global_stats(ms, gr_name):

    global_mean_ms = np.mean(ms)
    global_std_ms = np.std(ms)
    global_median_ms = np.median(ms)
    global_min_ms = min(ms)
    global_max_ms = max(ms)
    global_range_ms = global_max_ms - global_min_ms
    
    return {gr_name + '_global_mean': global_mean_ms,
            gr_name + '_global_std': global_std_ms,
            gr_name + '_global_median': global_median_ms,
            gr_name + '_global_min': global_min_ms,
            gr_name + '_global_max': global_max_ms,
            gr_name + '_global_range': global_range_ms }
            
#%% Computes Graph Measures from connectivity features (outdated)

def compute_graph_measures(fts):

    filenames = pd.read_excel('Metadata_train.xlsx')['Filename']
    ms_names = ['imcoh', 'plv', 'mi', 'pdc']
    bd_names = ['Global', 'Delta', 'Theta', 'Alpha', 'Beta']
    
    # dict to store all measures from all subjects
    graph_ms = {}
    
    for filename in filenames:
        # dict to store each subject's measures
        gr_ms_measure = {}
        
        for ms_name in ms_names:
            
            # MI does not have frequency bands
            if ms_name == 'mi':
                bd_names = ['Global']
            else:
                bd_names = ['Global', 'Delta', 'Theta', 'Alpha', 'Beta']
            
            # dict to store each band's measures
            gr_ms_bds = {}   
            for bd_name in bd_names:
                
                # get connectivity features
                ft = fts[ms_name][filename][bd_name]['Mean']
                
                # pdc is directed, no need to get symmetric matrix
                if ms_name != 'pdc':
                    # transform triangular matrix to symmetric square
                    ft = ft + ft.T - np.diag(np.diag(ft))
                  
                # imcoh may have negative values, let's consider abs only
                if ms_name == 'imcoh':
                    ft = abs(ft)
                
                # compute graphs measures
                global_efficiency = brainconn.distance.efficiency_wei(ft)
                betweenness_centr = brainconn.centrality.betweenness_wei(ft)
                
                bt_dict = _compute_global_stats(betweenness_centr, 'betweness_centr')
                
                # pdc is directed -> different functions
                if ms_name == 'pdc':
                    clustering_coef = brainconn.clustering.clustering_coef_wd(ft)
                    cc_dict = _compute_global_stats(clustering_coef, 'clustering_coef')
                    
                    in_strength, out_strength, _ = brainconn.degree.strengths_dir(ft)
                    in_dict = _compute_global_stats(in_strength, 'incoming_flow')
                    out_dict = _compute_global_stats(out_strength, 'outgoing_flow')
                    
                    # save in dict
                    gr_ms_bds[bd_name] = {
                        'clustering_coefficient': clustering_coef,
                        'global_efficiency': global_efficiency,
                        'betweness_centrality': betweenness_centr,
                        'incoming_flow': in_strength, 
                        'outgoing_flow': out_strength }
                    
                    stats = bt_dict
                    stats.update(cc_dict)
                    stats.update(in_dict)
                    stats.update(out_dict)
                    
                else:
                    clustering_coef = brainconn.clustering.clustering_coef_wu(ft)
                    cc_dict = _compute_global_stats(clustering_coef, 'clustering_coef')
                    
                    strengths = brainconn.degree.strengths_und(ft)
                    st_dict = _compute_global_stats(strengths, 'node_strengths')
                    
                    # save in dict
                    gr_ms_bds[bd_name] = {
                        'clustering_coefficient': clustering_coef,
                        'global_efficiency': global_efficiency,
                        'betweness_centrality': betweenness_centr,
                        'node_strengths': strengths }
                    
                    stats = bt_dict
                    stats.update(cc_dict)
                    stats.update(st_dict)
                    
                stats = pd.DataFrame.from_dict(stats, orient='index', 
                                               columns=[filename]).transpose()
                
                gr_ms_bds[bd_name]['stats'] = stats
                    
            gr_ms_measure[ms_name] = gr_ms_bds
        graph_ms[filename] = gr_ms_measure
        
    return graph_ms

#%% Subgroups Graph Measures

# Compute mean, std, median, ratio for left and right subgroups
def _compute_graph_mean_std(graph_ms, gr_n, conn_n, bd_n, sub_n, filename):

    m = np.mean(graph_ms)
    s = np.std(graph_ms)
    
    m_name = gr_n + '-' + conn_n + '-' + bd_n + '-' + sub_n + '-Mean'
    s_name = gr_n + '-' + conn_n + '-' + bd_n + '-' + sub_n + '-Std'

    return pd.DataFrame(data=[m,s], index=[m_name, s_name], columns=[filename])

# Final Computation
def compute_graph_subgroup_measures(fts):

    # def compute_subgroups_graph_measures(fts):
    filenames = pd.read_excel('Metadata_train.xlsx')['Filename']
    
    ms_names = ['imcoh', 'plv', 'mi', 'pdc']
    bd_names = ['Global', 'Delta', 'Theta', 'Alpha', 'Beta']
    
    # ch_names = ['Fz', 'Cz', 'Pz', 'Fp1', 'F3', 'C3', 'P3', 'O1','F7',
    #             'T3', 'T5', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6']
    
    subgroups = {
            'FR': ['Fp1', 'F7', 'T3', 'F3', 'C3', 'Fz', 'Cz'],
            'FL': ['Fp2', 'F8', 'T4', 'F4', 'C4', 'Fz', 'Cz'],
            'BR': ['T3', 'T5', 'O1', 'C3', 'P3', 'Cz', 'Pz'],
            'BL': ['T4', 'T6', 'O2', 'C4', 'P4', 'Cz', 'Pz'] }
            # 'R': ['Fz', 'Cz', 'Pz', 'Fp1', 'F7', 'F3', 'T3', 'C3', 'T5', 'P3', 'O1'],
            # 'L': ['Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'F8', 'C4', 'T4', 'P4', 'T6', 'O2'],
            # 'ALL': ch_names }
           
    subgroups_names = ['FR', 'FL', 'BR', 'BL']#, 'R', 'L', 'ALL']
    
    # dict to store all measures from all subjects
    graph_ms = {}
    
    for filename in filenames[0:10]:
        # df to store each subject's measures
        gr_ms = pd.DataFrame()
        
        for ms_name in ms_names:
            
            # MI does not have frequency bands
            if ms_name == 'mi':
                bd_names = ['Global']
            else:
                bd_names = ['Global', 'Delta', 'Theta', 'Alpha', 'Beta']
            
            for bd_name in bd_names:
                ft = fts[ms_name][filename][bd_name]
                
                for sub_n in subgroups_names:
                    chs = subgroups[sub_n]
                    ft_df_subgroup = _features_subgroup_combination(ft, chs, ms_name, imcohAbs=True)
                    ft_np_subgroup = ft_df_subgroup.to_numpy()
                
                    # efficiency
                    efficiency = brainconn.distance.efficiency_wei(ft_np_subgroup)
                    eff_name = 'efficiency' + '-' + ms_name + '-' + bd_name + '-' + sub_n
                    gr_df = pd.DataFrame(data=efficiency, index=[eff_name], columns=[filename])
                    gr_ms = pd.concat([gr_ms, gr_df], axis=0)
    
                    # betweenness centrality
                    betweeness = brainconn.centrality.betweenness_wei(ft_np_subgroup)
                    gr_df = _compute_graph_mean_std(betweeness, 'betweness_centr', ms_name, bd_name, sub_n, filename)
                    gr_ms = pd.concat([gr_ms, gr_df], axis=0)
                    
                    # pdc is directed -> different functions
                    if ms_name == 'pdc':
                        # clustering coefficient
                        clustering_coef = brainconn.clustering.clustering_coef_wd(ft_np_subgroup)
                        gr_df = _compute_graph_mean_std(clustering_coef, 'clustering_coef', ms_name, bd_name, sub_n, filename)
                        gr_ms = pd.concat([gr_ms, gr_df], axis=0)
                        
                        # incoming and outgoing flows
                        in_str, out_str, _ = brainconn.degree.strengths_dir(ft_np_subgroup)
                        gr_df = _compute_graph_mean_std(in_str, 'incoming_flow', ms_name, bd_name, sub_n, filename)
                        gr_ms = pd.concat([gr_ms, gr_df], axis=0)
                        gr_df = _compute_graph_mean_std(out_str, 'outgoing_flow', ms_name, bd_name, sub_n, filename)
                        gr_ms = pd.concat([gr_ms, gr_df], axis=0)
                                        
                    # conn measures not pdc undirected -> different functions
                    else:
                        # clustering cofficients
                        clustering_coef = brainconn.clustering.clustering_coef_wu(ft_np_subgroup)
                        gr_df = _compute_graph_mean_std(clustering_coef, 'clustering_coef', ms_name, bd_name, sub_n, filename)
                        gr_ms = pd.concat([gr_ms, gr_df], axis=0)
                        
                        # strengths
                        strength = brainconn.degree.strengths_und(ft_np_subgroup)
                        gr_df = _compute_graph_mean_std(strength, 'node_strengths', ms_name, bd_name, sub_n, filename)
                        gr_ms = pd.concat([gr_ms, gr_df], axis=0)
        
        # final dict with all sibjects
        graph_ms[filename] = gr_ms
    
    return graph_ms