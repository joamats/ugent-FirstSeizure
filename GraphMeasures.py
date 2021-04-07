import pandas as pd
import numpy as np
import brainconn
from DataPreparation import get_saved_features
from Pickle import createPickleFile

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
            
#%% Computes Graph Measures from connectivity features

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

#%% Run

# _, fts = get_saved_features(withGraphs=False)
# graph_ms = compute_graph_measures(fts)
# # createPickleFile(graph_ms, '../2_Features_Data/128Hz/' + 'graphMeasures')
                
                