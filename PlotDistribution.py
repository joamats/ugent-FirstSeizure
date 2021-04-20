from matplotlib import pyplot as plt
import seaborn as sb


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