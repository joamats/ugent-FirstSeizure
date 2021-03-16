from Pickle import getPickleFile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from matplotlib import pyplot

#%% 

#Computes the best features using Mutual Information
def select_features_mutual_info(X_train, y_train, X_test):
	# configure to select all features
	fs = SelectKBest(score_func=mutual_info_classif, k='all')
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

#Computes the best features using F-test
def select_features_f_test(X_train, y_train, X_test):
	# configure to select all features
	fs = SelectKBest(score_func=f_classif, k='all')
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

#Plots the best features based on MI or F-test method
#method is either 'f_test' or 'MI'
def select_features(X_train, y_train, X_test, method):
    if method=='MI':
        X_train, X_test, fs = select_features_mutual_info(X_train, y_train, X_test)
    # what are scores for the features
    for i in range(len(fs.scores_)):
    	print('Feature %d: %f' % (i, fs.scores_[i]))
    # plot the scores
    fig=pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
    pyplot.show()
    return fig


#%% Run 

datasets = getPickleFile('../ML_Data/' + 'datasets')