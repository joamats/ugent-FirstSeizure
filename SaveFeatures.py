import pandas as pd
from FeatureExtraction import extract_features
from PreProcessing import epochs_selection_bandpower
from Pickle import getPickleFile, createPickleFile

filenames = pd.read_excel('Metadata_train.xlsx')['Filename']

IMCOH = {}
PLV = {}
MI = {}
PDC = {}

# over all subjects
for i, filename in enumerate(filenames):
    saved_epochs = getPickleFile('../PreProcessed_Data/128Hz/' + filename)
    bd_names, s_epochs = epochs_selection_bandpower(saved_epochs)
    IMCOH[filename], PLV[filename], \
    MI[filename], PDC[filename] = extract_features(bd_names, s_epochs)
    
    # save features in pickle
    createPickleFile(IMCOH, '../Features/128Hz/' + 'imcoh')
    createPickleFile(PLV, '../Features/128Hz/' + 'plv')
    createPickleFile(MI, '../Features/128Hz/' + 'mi')
    createPickleFile(PDC, '../Features/128Hz/' + 'pdc')          
                