import seaborn as sb
import pandas as pd
import numpy as np
from BestRankedFeatures import best_ranked_features
from Pickle import getPickleFile

dataset = getPickleFile('../3_ML_Data/128Hz/dataset')
X_tr = dataset['X_tr']

fts_names = getPickleFile('../3_ML_Data/128Hz/featuresNames')

best_fts = best_ranked_features(dataset,fts_names, k_features=10)
best_idxs = best_fts.index

# filtered df with best features only
X_df = pd.DataFrame(data=X_tr[:,best_idxs], columns=best_fts['fts_names'])
corr_df = X_df.corr()

sb.heatmap(corr_df, annot=True, cmap="YlGnBu", annot_kws={"size": 25 / np.sqrt(len(corr_df))})
