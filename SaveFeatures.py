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
    saved_epochs = getPickleFile('../1_PreProcessed_Data/128Hz/' + filename)
    bd_names, s_epochs = epochs_selection_bandpower(saved_epochs)
    IMCOH[filename], PLV[filename], \
    MI[filename], PDC[filename] = extract_features(bd_names, s_epochs)
    
    # save features in pickle
    createPickleFile(IMCOH, '../2_Features_Data/128Hz/' + 'imcoh')
    createPickleFile(PLV, '../2_Features_Data/128Hz/' + 'plv')
    createPickleFile(MI, '../2_Features_Data/128Hz/' + 'mi')
    createPickleFile(PDC, '../2_Features_Data/128Hz/' + 'pdc')          
                