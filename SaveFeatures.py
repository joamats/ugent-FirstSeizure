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
for filename in filenames[[0]]:
    saved_epochs = getPickleFile('../PreProcessed_Data/' + filename)
    bd_names, s_epochs = epochs_selection_bandpower(saved_epochs)
    IMCOH[filename], PLV[filename], \
    MI[filename], PDC[filename] = extract_features(bd_names, s_epochs)
    
    # save features in pickle
    createPickleFile(IMCOH, '../Features/' + 'imcoh')
    createPickleFile(PLV, '../Features/' + 'plv')
    createPickleFile(MI, '../Features/' + 'mi')
    createPickleFile(PDC, '../Features/' + 'pdc')          
                