from Pickle import getPickleFile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

#%% Feature Selection functions


# Plots the best features based on MI or F-test method
# method is either 'f_test' or 'MI'
def select_features(X_train, y_train, X_test, method):
    
    if method == 'mi':
    # configure to select all features
        fs = SelectKBest(score_func=mutual_info_classif, k='all')
       
    elif method == 'anova':
        fs = SelectKBest(score_func=f_classif, k=50)
    
   	# learn relationship from training data
    fs.fit(X_train, y_train)
       
   	# transform train input data
    X_train_fs = fs.transform(X_train)
       
   	# transform test input data
    X_test_fs = fs.transform(X_test)
    
    # # what are scores for the features
    # for i in range(len(fs.scores_)):
    #     print('Feature %d: %f' % (i, fs.scores_[i]))
           
    # plot the scores
    fig = plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
    plt.show()
 
    idx = fs.get_support(indices=True)   
 
    return X_train_fs, X_test_fs, idx, fs, fig


#%% Run 

datasets = getPickleFile('../ML_Data/' + 'datasets')
fts_names = getPickleFile('../Features/' + 'featuresNames')
sbjs_names = getPickleFile('../Features/' + 'featuresNames')

#%%

X_tr = datasets[0]['train'][0]['X_tr']
y_tr = datasets[0]['train'][0]['y_tr']
X_val = datasets[0]['train'][0]['X_val']
y_val = datasets[0]['train'][0]['y_val']

norm_scaler = StandardScaler(with_mean=True, with_std=True)

X_tr = norm_scaler.fit_transform(X_tr)
X_val = norm_scaler.fit_transform(X_val) 

fig1 = plt.figure(1)

X_tr_fs, X_val_fs, idx, fs, fig1 = select_features(X_tr, y_tr, X_val, method='anova')
sl_fts_names = fts_names[idx]
print(sl_fts_names)



