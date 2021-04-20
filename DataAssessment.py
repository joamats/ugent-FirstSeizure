import pandas as pd
import seaborn as sb
import numpy as np
from matplotlib import pyplot as plt 

from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from Pickle import getPickleFile

#%% Data distribution - look for data imbalance

def plot_data_distribution(dataset, labels_names, mode):
    
    num_labels = len(labels_names) 
    y_tr = dataset['y_tr']
    
    counts = []
    
    for i in range(num_labels):
        counts.append(len(y_tr[y_tr == i]))
    
    s = sum(counts)
    r_counts = [i/s for i in counts]
    
    plt.figure()
    fig = sb.barplot(x=labels_names, y=r_counts, palette=sb.color_palette("hls", num_labels))
    plt.title(mode + ' Data Distribution', fontsize=10)
    
    return fig

#%% t-SNE plot - assess separability

def plot_tsne(dataset, labels_names, mode='Diagnosis'):
      
    num_labels = len(labels_names) 
    
    X_tr = dataset['X_tr']
    y_tr = dataset['y_tr']
    
    labels = []
    
    for y_i in y_tr:
        for i, lb in enumerate(labels_names):
            if y_i == i:
                labels.append(lb)
    
    tsne= TSNE(n_components=2, random_state=42, perplexity=30, early_exaggeration=12, learning_rate=200)
    X_embedded  = tsne.fit_transform(X_tr)
    df = pd.DataFrame()
    df['component1'] = X_embedded[:, 0]
    df['component2'] = X_embedded[:, 1]
    df['y'] = labels
    
    plt.figure()
    
    fig = sb.scatterplot(
        x="component1", y="component2",
        hue="y",
        s=15,
        palette=sb.color_palette("hls", num_labels),
        data=df,
        legend="full",
        alpha=1
    )
    fig.grid(False)
    plt.title(mode + ' Classification t-SNE', fontsize=10)
    legend = plt.legend(title='Labels', bbox_to_anchor=(1.3, 0.5), loc='center right', prop={'size': 6})
    plt.setp(legend.get_title(),fontsize='xx-small')
    plt.style.use('default') 
    
    return fig

#%% Feature selection with ANOVA - look for best ranked features

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

#%% Features Correlation Matrix

def fts_correlation_matrix(dataset, fts_names, k_features=None):

    X_tr = dataset['X_tr']

    if k_features is None:
        X_df = pd.DataFrame(data=X_tr, columns=fts_names)
        corr_df = X_df.corr()
                
    else:
        best_fts = best_ranked_features(dataset, fts_names, k_features)
        best_idxs = best_fts.index
        # filtered df with best features only
        X_df = pd.DataFrame(data=X_tr[:,best_idxs], columns=best_fts['fts_names'])
        
        corr_df = X_df.corr().round(decimals=2)
        
        plt.figure()
        # Show annotation corr only if 10 or less features are being showns
        if k_features <= 10:
            ax = sb.heatmap(corr_df, annot=True, cmap="Blues", xticklabels=True, yticklabels=True, linewidths=.5)
        elif k_features > 10:
            ax = sb.heatmap(corr_df, annot=False, cmap="Blues", xticklabels=True, yticklabels=True, linewidths=.5)
            
        ax.tick_params(axis='both', labelsize=5)
        plt.title('Features Correlation Matrix')
        ax.set_ylabel('')    
        ax.set_xlabel('')
    
    
    return corr_df