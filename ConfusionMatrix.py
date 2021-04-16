from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sb
import matplotlib.pyplot as plt

from Pickle import getPickleFile


def mlp_pca_wholedset_confusion_matrix(dataset):

    print('\nMLP + PCA\n')
    
    # Feature Normalization
    norm_scaler = StandardScaler(with_mean=True, with_std=True)
    minMax_scaler = MinMaxScaler()
    
    # MLP Model
    mlp = MLPClassifier(hidden_layer_sizes= (150,150), activation = 'relu',
                                             random_state=42, max_iter = 1000,
                                             learning_rate = 'adaptive',
                                             alpha=0.00001)
    
    # Dimensionality Reduction
    dim_red = PCA(n_components = 20, random_state=42)
    
    
    # Pipeline
    clf = Pipeline(steps=[('norm_scaler',norm_scaler),
                                ('min_max', minMax_scaler),
                                ('dim_red', dim_red),
                                ('classifier', mlp)])
    
    X_tr = dataset['X_tr']
    y_tr = dataset['y_tr']
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2)
    
    clf.fit(X_tr, y_tr)
    y_predict= clf.predict(X_val)
    
    confusionMatrix= confusion_matrix(y_val, y_predict)
    
    return confusionMatrix

dataset = getPickleFile('../3_ML_Data/128Hz/dataset')
fts_names = getPickleFile('../3_ML_Data/128Hz/featuresNames')
labels_names = getPickleFile('../3_ML_Data/128Hz/labelsNames')

# MLP + PCA
confusionMatrix = mlp_pca_wholedset_confusion_matrix(dataset)
#%%
sb.heatmap(confusionMatrix, annot=True, cmap='Blues')
plt.title('Confusion Matrix Diagnosis Classification (MLP + PCA - No bandpower)')
plt.xlabel('Target Class')
plt.ylabel('Predicted Class')


