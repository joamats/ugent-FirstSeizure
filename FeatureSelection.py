from Pickle import getPickleFile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt


#%% Feature Selection functions


# Plots the best features based on MI or F-test method
# method is either 'f_test' or 'MI'
def select_features(X_train, y_train, X_test, method):
    
    if method == 'mi':
    # configure to select all features
        fs = SelectKBest(score_func=mutual_info_classif, k=150)
       
    elif method == 'anova':
        fs = SelectKBest(score_func=f_classif, k=150)
    
   	# learn relationship from training data
    fs.fit(X_train, y_train)
       
   	# transform train input data
    X_train_fs = fs.transform(X_train)
           
    # # what are scores for the features
    # for i in range(len(fs.scores_)):
    #     print('Feature %d: %f' % (i, fs.scores_[i]))
           
    # plot the scores
    fig = plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
    plt.hlines(y=9.341803308, xmin=0, xmax=8395, linestyles='dashed')
    plt.xlabel('Features')
    plt.ylabel('Anova Score')
    plt.title('Feature Selection')
    plt.show()
 
    idx = fs.get_support(indices=True)   
 
    return X_train_fs, idx, fs, fig


#%% Run 

datasets = getPickleFile('../3_ML_Data/' + 'datasets')
fts_names = getPickleFile('../3_ML_Data/' + 'featuresNames')

#%%

X_tr = datasets[0]['X_tr']
y_tr = datasets[0]['y_tr']

norm_scaler = StandardScaler(with_mean=True, with_std=True)
minMax_scaler = MinMaxScaler()

X_tr = norm_scaler.fit_transform(X_tr)
X_tr = minMax_scaler.fit_transform(X_tr)

# Feature Selection
_k_features = 200

selector = SelectKBest(score_func=f_classif,
                        k=_k_features)
X_tr=selector.fit_transform(X_tr, y_tr)

fig1 = plt.figure(1)

X_tr_fs, idx, fs, fig1 = select_features(X_tr, y_tr,  method='anova')


