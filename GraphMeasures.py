import pandas as pd
import numpy as np
import brainconn
from FeaturePreparation import get_saved_features
from Pickle import createPickleFile

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
                
                # compute graphs measures
                global_efficiency = brainconn.distance.efficiency_wei(ft)
                betweenness_centr = brainconn.centrality.betweenness_wei(ft)
                
                # pdc is directed -> different functions
                if ms_name == 'pdc':
                    clustering_coef = brainconn.clustering.clustering_coef_wd(ft)
                    in_strength, out_strength, _ = brainconn.degree.strengths_dir(ft)
                    # save in dict
                    gr_ms_bds[bd_name] = {
                        'clustering_coefficient': clustering_coef,
                        'global_efficiency': global_efficiency,
                        'betweness_centrality': betweenness_centr,
                        'incoming_flow': in_strength, 
                        'outgoing_flow': out_strength }
                else:
                    clustering_coef = brainconn.clustering.clustering_coef_wu(ft)
                    strengths = brainconn.degree.strengths_und(ft)
                    # save in dict
                    gr_ms_bds[bd_name] = {
                        'clustering_coefficient': clustering_coef,
                        'global_efficiency': global_efficiency,
                        'betweness_centrality': betweenness_centr,
                        'node_strengths': strengths }
                    
            gr_ms_measure[ms_name] = gr_ms_bds
        graph_ms[filename] = gr_ms_measure
        
    return graph_ms

#%% Run

fts = get_saved_features(withGraphs=False)
graph_ms = compute_graph_measures(fts)
createPickleFile(graph_ms, '../Features/' + 'graphMeasures')
                
                