from Pickle import getPickleFile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt
import pandas as pd

#%% Run 

datasets = getPickleFile('../3_ML_Data/128Hz/' + 'datasets')
fts_names = getPickleFile('../3_ML_Data/128Hz/' + 'featuresNames')

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
X_tr = selector.fit_transform(X_tr, y_tr)

idx = selector.get_support(indices=True)
scores = selector.scores_
a = pd.DataFrame(data=scores[idx], index=idx, columns=['score'])
a['fts_names'] = fts_names[idx]
a = a.sort_values(by='score', ascending=False)


# fts_selected = pd.DataFrame(data=)
# fig1 = plt.figure(1)

# plot the scores
# fig = plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
# plt.hlines(y=9.341803308, xmin=0, xmax=8395, linestyles='dashed')
# plt.xlabel('Features')
# plt.ylabel('Anova Score')
# plt.title('Feature Selection')
# plt.show()
 
# idx = fs.get_support(indices=True)  


