from matplotlib import pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd

#Example of modes_list:
#MODE=["AntecedentFamilyEpileptic", "AntecedentFamilyNonEpileptic","AntecedentFamilyOther"]
def several_modes_data_and_labels(modes_list):
    datasets=[]
    labels_names_list=[]
    for i in range(3):
    
        labels, filenames = get_filenames_labels(mode=MODE[i])
        # Make array
        data = make_features_array(filenames, bdp_ms, conn_ms, gr_ms, asy_ms)
        fts_names = data.columns
        
        createPickleFile(data, '../2_Features_Data/128Hz/' + 'allFeatures')
        createPickleFile(fts_names, '../3_ML_Data/128Hz/' + 'featuresNames')
        
        labels_names = add_labels_to_data_array(data, labels, mode=MODE[i])
        labels_names_list.append(labels_names)
        dataset = dataset_split(data)
        datasets.append(dataset)
        
    createPickleFile(datasets, '../3_ML_Data/128Hz/' + 'dataset')
    createPickleFile(labels_names_list, '../3_ML_Data/128Hz/' + 'labelsNames')
    return labels_names_list, datasets

def plot_data_distribution(dataset, labels_names, mode, title=None):
    
    #Non-normalized multiple countplot
    # if isinstance(dataset, list):
    #     dictionary={}
    #     for i in range(np.size(dataset)):
    #         y_tr=(dataset[i]['y_tr'])
    #         dictionary[mode[i]]=y_tr
            
    #     dframe=pd.DataFrame.from_dict(dictionary, orient='index')
    #     dframe=dframe.transpose()
    #     df=pd.melt(dframe, value_vars=mode) 
        
    #     fig = sb.catplot(x="variable", hue="value", aspect=1.5, data=df, kind ='count')
    #     plt.title(title + ' Data Distribution', fontsize=10)
    #     plt.tight_layout()
    
    #Normalized multiple countplot    
    if isinstance(dataset, list):
        dictionary={"Diagnosis":['Non-Epileptic', 'Epileptic']}
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
    
    return fig, dframe, df