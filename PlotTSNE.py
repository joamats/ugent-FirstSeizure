from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt 

def plot_tsne(dataset, labels_names, mode='Diagnosis'):
      
    num_labels = len(labels_names) 
    
    X_tr = dataset['X_tr']
    y_tr = dataset['y_tr']
    
    labels = []
    
    for y_i in y_tr:
        for i, lb in enumerate(labels_names):
            if y_i == i:
                labels.append(lb)
    
    X_embedded = TSNE(n_components=2, random_state=42).fit_transform(X_tr)
         
    df = pd.DataFrame()
    df['component1'] = X_embedded[:, 0]
    df['component2'] = X_embedded[:, 1]
    df['y'] = labels
    
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