from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_roc_curve, classification_report
import seaborn as sb
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler

from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd


from Pickle import getPickleFile

def train_validation_no_cv(dataset, labels_names):

    # Feature Normalization
    norm_scaler = StandardScaler(with_mean=True, with_std=True)
    # minMax_scaler = MinMaxScaler()

    # # MLP Model
    # mlp = MLPClassifier(hidden_layer_sizes= (150,150), activation = 'relu',
    #                                          random_state=42, max_iter = 1000,
    #                                          learning_rate = 'adaptive',
    #                                          alpha=0.00001)
    
    svc = SVC(C=1, gamma=0.1, kernel='sigmoid', random_state=42)

    # Dimensionality Reduction
    # dim_red = PCA(n_components = 20, random_state=42)
    dim_red = SelectKBest(k=25, score_func=f_classif)

    # Pipeline
    clf = Pipeline(steps=[('norm_scaler',norm_scaler), 
                                ('dim_red', dim_red),
                                ('classifier', svc)]) #('min_max', minMax_scaler),

    X_tr = dataset['X_tr']
    y_tr = dataset['y_tr']
    
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2)
    rus = RandomUnderSampler(random_state=42)
    X_tr, y_tr = rus.fit_resample(X_tr, y_tr)

    clf.fit(X_tr, y_tr)
    y_predict= clf.predict(X_val)

    confusionMatrix= confusion_matrix(y_val, y_predict)
    plt.figure()
    plot_roc_curve(estimator=clf, X=X_val, y=y_val)
    plt.title('Sleep Diagnosis Classification (SVM + ANOVA) ROC_AUC')
    rep = classification_report(y_val, y_predict, target_names=labels_names, zero_division=0)

    return X_tr, clf, rep, confusionMatrix

dataset = getPickleFile('../3_ML_Data/128Hz/dataset')
fts_names = getPickleFile('../3_ML_Data/128Hz/featuresNames')
labels_names = getPickleFile('../3_ML_Data/128Hz/labelsNames')


X_tr, clf, rep, confusionMatrix = train_validation_no_cv(dataset, labels_names)
#%%
plt.figure()
sb.heatmap(confusionMatrix, annot=True, cmap='Blues')
plt.title('Confusion Matrix Sleep Diagnosis Classification (SVM + ANOVA)')
plt.xlabel('Target Class')
plt.ylabel('Predicted Class')

print(rep)