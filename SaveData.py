from Pickle import createPickleFile
from FeaturePreparation import get_saved_features,  make_features_array, \
                            add_labels_to_data_array, dataset_split

conn_ms, graph_ms = get_saved_features(withGraphs=True)
data = make_features_array(conn_ms, graph_ms)
fts_names = data.columns

createPickleFile(data, '../Features/' + 'allFeatures')
createPickleFile(fts_names, '../ML_Data/' + 'featuresNames')

add_labels_to_data_array(data)
datasets = dataset_split(data)

createPickleFile(datasets, '../ML_Data/' + 'datasets')