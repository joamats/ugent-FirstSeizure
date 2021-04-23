import pandas as pd
import numpy as np
import brainconn
from DataPreparation import get_saved_features
from FeatureExtraction import _features_subgroup_combination

#%% Subgroups Graph Measures

# Compute mean, std, median, ratio for left and right subgroups
def _compute_graph_mean_std(graph_ms, gr_n, conn_n, bd_n, sub_n, filename):

    if gr_n == 'incoming_flow':
       
        maxi = np.max(graph_ms)
        maxi_name = gr_n + '-' + conn_n + '-' + bd_n + '-' + sub_n + '-Max'
        
        m = np.mean(graph_ms)
        gr_n = 'node_strengths'
        m_name = gr_n + '-' + conn_n + '-' + bd_n + '-' + sub_n + '-Mean'
        
        d_list = [m, maxi]
        n_list = [m_name, maxi_name]
    
    elif gr_n == 'outgoing_flow':
        d_list = [np.max(graph_ms)]       
        n_list = [gr_n + '-' + conn_n + '-' + bd_n + '-' + sub_n + '-Max']
    
    else:
        
        m = np.mean(graph_ms)
        s = np.std(graph_ms)
        d_list = [m, s]
    
        m_name = gr_n + '-' + conn_n + '-' + bd_n + '-' + sub_n + '-Mean'
        s_name = gr_n + '-' + conn_n + '-' + bd_n + '-' + sub_n + '-Std'
        n_list = [m_name, s_name]

    return pd.DataFrame(data=d_list, index=n_list, columns=[filename])

# Final Computation
def compute_graph_subgroup_measures(fts, bipolar=True):

    # def compute_subgroups_graph_measures(fts):
    filenames = pd.read_excel('Metadata_train.xlsx')['Filename']
    
    ms_names = ['imcoh', 'plv', 'mi', 'pdc']
    bd_names = ['Global', 'Delta', 'Theta', 'Alpha', 'Beta']
    
    if bipolar:
        subgroups = {
                'FR': ['Fp1-F3', 'F3-C3', 'Fp1-F7', 'F7-T3', 'Fz-Cz'],
                'FL': ['Fp2-F4', 'F4-C4', 'Fp2-F8', 'F8-T4', 'Fz-Cz'],
                'BR': ['T3-T5', 'T5-O1', 'C3-P3', 'P3-O1', 'Cz-Pz'],
                'BL': ['T4-T6', 'T6-O2', 'C4-P4', 'P4-O2', 'Cz-Pz'] }
    
    else:
        subgroups = {
                'FR': ['Fp1', 'F7', 'T3', 'F3', 'C3', 'Fz', 'Cz'],
                'FL': ['Fp2', 'F8', 'T4', 'F4', 'C4', 'Fz', 'Cz'],
                'BR': ['T3', 'T5', 'O1', 'C3', 'P3', 'Cz', 'Pz'],
                'BL': ['T4', 'T6', 'O2', 'C4', 'P4', 'Cz', 'Pz'] }
        

    
    subgroups_names = ['FR', 'FL', 'BR', 'BL']
    
    # dict to store all measures from all subjects
    graph_ms = {}
    
    for filename in filenames:
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
                    ft_df_subgroup = _features_subgroup_combination(ft, chs, ms_name)
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