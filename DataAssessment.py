import pandas as pd
import seaborn as sb
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import axes as ax

from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from Pickle import getPickleFile

#%% Data distribution - look for data imbalance

def plot_data_distribution(dataset, labels_names, mode, title=None, xlabel=None, ylabel=None, xtickslabels=None):
    
    # #Non-normalized multiple countplot
    # if isinstance(dataset, list):
    #     dictionary={}
    #     for i in range(np.size(dataset)):
    #         y_tr=(dataset[i]['y_tr'])
    #         dictionary[mode[i]]=y_tr
            
    #     dframe=pd.DataFrame.from_dict(dictionary, orient='index')
    #     dframe=dframe.transpose()
    #     df=pd.melt(dframe, value_vars=mode)
        
    #     fig = sb.catplot(x="variable", hue="value", legend=False, aspect=1.5, data=df, kind ='count')
    #     plt.title(title + ' Data Distribution', fontsize=10)
    #     plt.tight_layout()
    #     plt.ylabel(ylabel)
    #     plt.xlabel(xlabel)
    #     plt.xticks(range(0,np.size(dataset)),xtickslabels)
    #     plt.legend(labels=labels_names[0], loc=1)
    
    
    #Normalized multiple countplot    
    if isinstance(dataset, list):
        dictionary={"Diagnosis":labels_names[0]}
        for i in range(np.size(dataset)):
            num_labels = len(labels_names[i]) 
            y_tr=(dataset[i]['y_tr'])
            counts = []
        
            for k in range(num_labels):
                counts.append(len(y_tr[y_tr == k]))
            
            s = sum(counts)
            r_counts = [j/s for j in counts]
            dictionary[mode[i]]=np.array(r_counts)
        dframe=pd.DataFrame.from_dict(dictionary, orient='index')
        dframe=dframe.transpose()
        df=pd.melt(dframe, id_vars=["Diagnosis"], value_vars=mode) 
        
        fig = sb.catplot(x="variable", y="value", hue="Diagnosis", aspect=1.5,
                          data=df, kind ='bar', legend=False, palette=sb.color_palette("hls", num_labels))
        plt.title(title + ' Data Distribution', fontsize=10)
        plt.tight_layout()
        plt.ylim(0,1)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.xticks(range(0,np.size(dataset)),xtickslabels)
        plt.legend(loc=1)
        
    else:
        num_labels = len(labels_names) 
        y_tr = dataset['y_tr']
        
        counts = []
        
        for i in range(num_labels):
            counts.append(len(y_tr[y_tr == i]))
        
        s = sum(counts)
        r_counts = [i/s for i in counts]
        
        fig = sb.barplot(x=labels_names, y=r_counts, palette=sb.color_palette("hls", num_labels))
        plt.title(mode + ' Data Distribution', fontsize=10)
        plt.ylim(0,1)
    
    return fig

# Plot Data Distribution

# # Plot Data Distribution for Family Antecedent
# fig_data_dist = plot_data_distribution(datasets, labels_names_list, MODE,
#                                        title="Family Antecedent Absolute",
#                                        xlabel="Family Antecedent",
#                                        ylabel="Absolute Distribution",
#                                        xtickslabels=['Epileptic', 'Non Epileptic', 'Other'])


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
    
    tsne= TSNE(n_components=2, random_state=42, perplexity=10, early_exaggeration=12, learning_rate=200)
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

def _best_fts(selector, fts_names):
    idx = selector.get_support(indices=True)
    scores = selector.scores_
    best_fts = pd.DataFrame(data=scores[idx], index=idx, columns=['score'])
    best_fts['fts_names'] = fts_names[idx]
    
    return best_fts.sort_values(by='score', ascending=False)

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
    
    return _best_fts(selector, fts_names)

#%% Histograms of best ranked features types

def count_best_fts_types(best_fts, MODE):

    fts_split = []
    fts_names = best_fts['fts_names'].unique()
    for fts in fts_names:
        fts_split.append(fts.split('-'))
        
    bst_type_1, bst_type_2, bst_type_3, = [], [], []
    fts_type_1 = ['bdp', 'imcoh', 'plv', 'mi', 'pdc']
    fts_type_2 = ['Global', 'Alpha', 'Theta', 'Delta', 'Beta']
    
    # Auxiliar for bst_type_3
    fts_type_list_conn = ['imcoh', 'plv', 'mi', 'pdc']
    fts_type_list_graph = ['betweness_centr', 'clustering_coef', 'incoming_flow',
                          'outgoing_flow', 'node_strengths', 'efficiency']
    
    for fts in fts_split:
        bst_type_1.append([i for i in fts_type_1 if i in fts][0])
        bst_type_2.append([i for i in fts_type_2 if i in fts][0])
        
        conn = [i for i in fts_type_list_conn if i in fts[0]]
        graph = [i for i in fts_type_list_graph if i in fts[0] and 'vs' not in fts[3]]
        asymmetry = [i for i in fts_type_list_graph if i in fts[0] and 'vs' in fts[3]]
        
        if conn != []:
            bst_type_3.append('Functional\nConnectivity')
        elif fts[0]=='bdp':
            bst_type_3.append('Bandpowers')
        elif graph != []:
            bst_type_3.append('Graph\nMeasures')
        elif asymmetry != []:
            bst_type_3.append('Asymmetry\nRatios')
        
    df_1 = pd.DataFrame(bst_type_1, columns=['Bandpower and Functional Connectivity'])
    df_2 = pd.DataFrame(bst_type_2, columns=['Frequency Band'])
    df_3 = pd.DataFrame(bst_type_3, columns=['Type of Measure'])
    
    # Initialize figure
    plt.figure()
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18,5), sharey=True)
    
    sb.countplot(x='Bandpower and Functional Connectivity', data=df_1, ax=axs[0]).set(xlabel=None)
    axs[0].title.set_text('Bandpower and Functional Connectivity')
    sb.countplot(x='Frequency Band', data=df_2, ax=axs[1]).set(xlabel=None)
    axs[1].title.set_text('Frequency Band')
    sb.countplot(x='Type of Measure', data=df_3, ax=axs[2]).set(xlabel=None)
    axs[2].title.set_text('Type of Measure')
    
    plt.suptitle(MODE + ' Most Selected Features through 5-Fold CV', fontsize=18, va='center')
    

#%% Features Correlation Matrix

def _filter_features(X_df, ms_keep, ms_exclude):
    # iterate over ms_keep to keep only its strings
    for ms in ms_keep:
        X_df = X_df.filter(regex=ms, axis=1)
    # iterate over ms_exclude to keep only its strings
    for ms in ms_exclude:
        X_df = X_df[X_df.columns.drop(list(X_df.filter(regex=ms)))]
        
    return X_df

def fts_correlation_matrix(dataset, fts_names, ms_keep=[], ms_exclude=[], k_best_features=0):
    # ms_keep is a list with str keywords of features to keep in this matrix
    # ms_exclude is a list with str keywords of features to remove
    # len(ms_keep) + len(ms_exclude) == 0 means no filter is done at all (both are [])
    # k_best_features will define that only highest ranked features (p-value) are considered

    X_tr = dataset['X_tr']

    # all features mode
    if k_best_features == 0 and len(ms_keep) + len(ms_exclude) == 0:
        X_df = pd.DataFrame(data=X_tr, columns=fts_names)
        return X_df.corr()
    
    # k best features mode
    if len(ms_keep) + len(ms_exclude) == 0 and k_best_features > 0:
        best_fts = best_ranked_features(dataset, fts_names, k_best_features)
        best_idxs = best_fts.index
        # filtered df with best features only
        X_df = pd.DataFrame(data=X_tr[:,best_idxs], columns=best_fts['fts_names'])
    
    # filter mode, based on methodological similarities
    elif len(ms_keep) + len(ms_exclude) > 0 and k_best_features == 0:
        X_df = pd.DataFrame(data=X_tr, columns=fts_names)
        X_df = _filter_features(X_df, ms_keep, ms_exclude)
                    
    else:
        raise AttributeError('Only one mode is possible: either you set filtering conditions, or k best features to be shown in correlation matrix. If ms_keep or ms_exclude are defined, k_features must not be simultaneously defined, and vice-versa.')
      
    # build corr matrix, rounded to 2 decimals
    corr_df = X_df.corr().round(decimals=2)
    # actual number of features considered
    num_fts = corr_df.shape[0]
       
    # plotting     
    plt.figure()
    # Show annotation corr only if 10 or less features are being showns
    if num_fts <= 10:
        ax = sb.heatmap(corr_df, annot=True, cmap="Blues", xticklabels=True, yticklabels=True, linewidths=.5)
    elif num_fts > 10:
        ax = sb.heatmap(corr_df, annot=False, cmap="Blues", xticklabels=True, yticklabels=True, linewidths=.5)
        
    ax.tick_params(axis='both', labelsize=5)
    plt.title('Features Correlation Matrix')
    ax.set_ylabel('')    
    ax.set_xlabel('')
    
    return corr_df

#%% Most and Least correlated features

def _get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def _get_top_abs_correlations(df, n=-1, ascending=False):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = _get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=ascending)
    if n == -1:
        return au_corr
    elif n > 0:
        return au_corr[0:n]

def most_least_correlated_fts(dataset, fts_names, n=10, ms_keep=[], ms_exclude=[]):
    X_tr = dataset['X_tr']
    X_df = pd.DataFrame(data=X_tr, columns=fts_names)
    
    if len(ms_keep) + len(ms_exclude) > 0:
        X_df = _filter_features(X_df, ms_keep, ms_exclude)
        
    corr_most =_get_top_abs_correlations(X_df, n, ascending=False)
    corr_least =_get_top_abs_correlations(X_df, n, ascending=True)
    
    return corr_most, corr_least

# Most and Least Correlated Features
# import seaborn as sb
# from matplotlib import pyplot as plt
# corr_most, corr_least = most_least_correlated_fts(dataset, fts_names, n=-1)
# plt.figure()
# sb.histplot(x=corr_most.values)
# plt.title('Pairs of features correlations distribution')
# plt.xlabel('Correlation')
