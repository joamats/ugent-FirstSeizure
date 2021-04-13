from sklearn.manifold import TSNE
from Pickle import getPickleFile
import pandas as pd
import seaborn as sb

def plot_tsne(dataset, num_labels=2):

    X_tr = dataset['X_tr']
    y_tr = dataset['y_tr']
    
    X_embedded = TSNE(n_components=2).fit_transform(X_tr)
         
    df = pd.DataFrame()
    df['one'] = X_embedded[:,0]
    df['two'] = X_embedded[:,1]
    df['y'] = y_tr
    
    fig = sb.scatterplot(
        x="one", y="two",
        hue="y",
        palette=sb.color_palette("hls", num_labels),
        data=df,
        legend="full",
        alpha=1
    )
    
    return fig
