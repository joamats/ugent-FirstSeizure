import pandas as pd
from Pickle import getPickleFile
import seaborn as sb
from matplotlib import pyplot as plt

filenames = pd.read_excel('Metadata_train.xlsx')['Filename']

graph_ms = getPickleFile('../Features/' + 'graphMeasures')

ch_names_original = ['Fz', 'Cz', 'Pz', 'Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T7', 
                          'P7', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T8', 'P8']
#New channel order
ch_names_new = ['Fp1', 'Fp2', 'F8', 'F4', 'Fz', 'F3', 'F7', 'T7', 'C3', 'Cz',
                'C4', 'T8', 'P8', 'P4', 'Pz', 'P3', 'P7', 'O1', 'O2']
ch_indexes = []

#New channel order's indexes on the ch_names_original variable
for ch in ch_names_new:
    ch_indexes.append(ch_names_original.index(ch))

spectral_feature = ['imcoh', 'mi', 'pdc', 'plv']
    
    
for filename in filenames[[201]]:
    for sp in spectral_feature:
        if sp == 'mi':
            bd_names = ['Global']
            graph_measures = ['betweness_centrality', 'clustering_coefficient',
            'global_efficiency', 'node_strengths']
            
        elif sp == 'pdc':
            graph_measures = ['betweness_centrality', 'clustering_coefficient',
                    'global_efficiency', 'incoming_flow', 'outgoing_flow']
            bd_names = ['Global', 'Delta', 'Theta', 'Alpha', 'Beta']
            
        else:
            graph_measures = ['betweness_centrality', 'clustering_coefficient',
            'global_efficiency', 'node_strengths']
            bd_names = ['Global', 'Delta', 'Theta', 'Alpha', 'Beta']
            
        for bd in bd_names:
            for gfm in graph_measures:
                if gfm != 'global_efficiency':
                    graph_ms[filename][sp][bd][gfm]=graph_ms[filename][sp][bd][gfm].reshape(-1,1)
                    graph_ms[filename][sp][bd][gfm]=graph_ms[filename][sp][bd][gfm][ch_indexes,:]
                    sb.scatterplot(x=ch_names_new, y=list(graph_ms[filename][sp][bd][gfm]))
                    plt.figure()
