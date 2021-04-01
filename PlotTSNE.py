from sklearn.manifold import TSNE
from Pickle import getPickleFile
import pandas as pd
import seaborn as sb

datasets = getPickleFile('../3_ML_Data/' + 'datasets')

X_tr = datasets[0]['X_tr']
y_tr = datasets[0]['y_tr']

X_embedded = TSNE(n_components=2).fit_transform(X_tr)
     
df = pd.DataFrame()
df['one'] = X_embedded[:,0]
df['two'] = X_embedded[:,1]
df['y'] = y_tr

sb.scatterplot(
    x="one", y="two",
    hue="y",
    palette=sb.color_palette("hls", 2),
    data=df,
    legend="full",
    alpha=0.8
)
