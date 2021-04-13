from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

def best_ranked_features(dataset, fts_names, k_features=200):

    X_tr = dataset['X_tr']
    y_tr = dataset['y_tr']
    
    norm_scaler = StandardScaler(with_mean=True, with_std=True)
    minMax_scaler = MinMaxScaler()
    
    X_tr = norm_scaler.fit_transform(X_tr)
    X_tr = minMax_scaler.fit_transform(X_tr)
    
    # Feature Selection
    selector = SelectKBest(score_func=f_classif,
                            k=k_features)
    X_tr = selector.fit_transform(X_tr, y_tr)
    
    idx = selector.get_support(indices=True)
    scores = selector.scores_
    best_fts = pd.DataFrame(data=scores[idx], index=idx, columns=['score'])
    best_fts['fts_names'] = fts_names[idx]

    return best_fts.sort_values(by='score', ascending=False)


