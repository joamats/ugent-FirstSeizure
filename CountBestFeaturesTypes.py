import seaborn as sns
from pandas import DataFrame
from matplotlib import pyplot as plt

fts_split=[]
fts_names=best_fts['fts_names']
for fts in fts_names:
    fts_split.append(fts.split('-'))
    
best_fts_type=[]
# fts_type_list=['bdp', 'imcoh', 'plv', 'mi', 'pdc']
fts_type_list=['Global', 'Alpha', 'Theta', 'Delta', 'Beta']
# fts_type_list_conn=['imcoh', 'plv', 'mi', 'pdc']
# fts_type_list_graph=['betweness_centr', 'clustering_coef', 'incoming_flow',
#                      'outgoing_flow', 'node_strengths', 'efficiency']

for fts in fts_split:
    best_fts_type.append([i for i in fts_type_list if i in fts][0])
    
# for fts in fts_split:
#     conn=[i for i in fts_type_list_conn if i in fts[0]]
#     graph=[i for i in fts_type_list_graph if i in fts[0] and 'vs' not in fts[3]]
#     asymmetry=[i for i in fts_type_list_graph if i in fts[0] and 'vs' in fts[3]]
#     if conn!=[]:
#         best_fts_type.append('Functional\nConnectivity')
#     elif fts[0]=='bdp':
#         best_fts_type.append('Bandpowers')
#     elif graph!=[]:
#         best_fts_type.append('Graph\nMeasures')
#     elif assimetry!=[]:
#         best_fts_type.append('Asymmetry\nRatios')
    
df = DataFrame (best_fts_type,columns=['Frequency Band'])

fig = sns.countplot(x="Frequency Band", data=df)
plt.title('Frequency Bands')


    