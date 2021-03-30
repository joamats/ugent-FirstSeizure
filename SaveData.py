from Pickle import createPickleFile
from DataPreparation import get_saved_features,  make_features_array, \
                            add_labels_to_data_array, dataset_split

conn_ms, graph_ms = get_saved_features(withGraphs=True)

#%% No Std

for ms in conn_ms:
    for subj in conn_ms[ms]:
        for bd in conn_ms[ms][subj]:
            del conn_ms[ms][subj][bd]['Std']

#%%
data = make_features_array(conn_ms, graph_ms, std = False)
fts_names = data.columns

createPickleFile(data, '../Features/' + 'allFeatures_noStd')
createPickleFile(fts_names, '../ML_Data/' + 'featuresNames_noStd')

add_labels_to_data_array(data)
datasets = dataset_split(data)

createPickleFile(datasets, '../ML_Data/' + 'datasets_noStd')