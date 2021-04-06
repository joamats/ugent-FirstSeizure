import pandas as pd
import numpy as np
import brainconn
from Pickle import createPickleFile, getPickleFile
from DataPreparation import get_saved_features

#%% Auxiliary Functions

# Build DataFrame from Numpy Array, with proper labels
def _build_conn_df(data_array):
    ch_names_original = ['Fz', 'Cz', 'Pz', 'Fp1', 'F3', 'C3', 'P3', 'O1','F7',
                         'T3', 'T5', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6']
    
    return pd.DataFrame(data_array, index=ch_names_original, columns=ch_names_original)

# Filter built DataFrame, for a subgroup of electrodes
def _get_subgroup_connectivity(conn_df, ch_names_subgroup):
    
    return conn_df.filter(items=ch_names_subgroup, axis=1).filter(items=ch_names_subgroup, axis=0)
    

# Compute ratio between subgroups
def _compute_ratio(a, b):
    
    max_val = max(abs(a), abs(b))
    min_val = min(abs(a), abs(b))
    
    if min_val == 0:
        return 0
    else:
        return max_val / min_val

# Compute mean, std, median, ratio for left and right subgroups
def _compute_stats(ms_left, ms_right, gr_name):

    right_mean_ms = np.mean(ms_right)
    left_mean_ms = np.mean(ms_left)
    right_std_ms = np.std(ms_right)
    left_std_ms = np.std(ms_left)
    right_median_ms = np.median(ms_right)
    left_median_ms = np.median(ms_left)
    
    ms_mean_left_vs_right = _compute_ratio(left_mean_ms, right_mean_ms)
    ms_std_left_vs_right = _compute_ratio(left_std_ms, right_std_ms)
    ms_median_left_vs_right = _compute_ratio(left_median_ms, right_median_ms)
    
    return {gr_name + '_right_mean': right_mean_ms,
            gr_name + '_left_mean': left_mean_ms,
            gr_name + '_right_std': right_std_ms,
            gr_name + '_left_std': left_std_ms,
            gr_name + '_right_median': right_median_ms,
            gr_name + '_left_median': left_median_ms,
            gr_name + '_mean_left_vs_right': ms_mean_left_vs_right,
            gr_name + '_std_left_vs_right': ms_std_left_vs_right,
            gr_name + '_median_left_vs_right': ms_median_left_vs_right}
            

#%% Local Efficiencies for Left and Right Electrode Groups

def compute_asymmetric_efficiencies(fts):
    filenames = pd.read_excel('Metadata_train.xlsx')['Filename']
    
    ms_names = ['imcoh', 'plv', 'mi', 'pdc']
    bd_names = ['Global', 'Delta', 'Theta', 'Alpha', 'Beta']
    
    # dict to store all measures from all subjects
    asymmetry_ms = {}
    
    for filename in filenames[[0]]:
        # dict to store each subject's measures
        gr_ms_measure = {}
        
        for ms_name in ms_names:
            
            # MI does not have frequency bands
            if ms_name == 'mi':
                bd_names = ['Global']
            else:
                bd_names = ['Global', 'Delta', 'Theta', 'Alpha', 'Beta']
            
            # dict to store each band's measures
            gr_ms_bds = {}   
            for bd_name in bd_names:
                
                # get connectivity features
                ft = fts[ms_name][filename][bd_name]['Mean']
                # build dataframe
                conn_df = _build_conn_df(ft)
                
                # right group
                right_group = ['Fz', 'Cz', 'Pz', 'Fp1', 'F7', 'F3', 'T3', 'C3', 'T5', 'P3', 'O1']
                conn_subgroup_right = _get_subgroup_connectivity(conn_df, right_group)
                ft_right = conn_subgroup_right.to_numpy()
    
                # left group
                left_group = ['Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'F8', 'C4', 'T4', 'P4', 'T6', 'O2']
                conn_subgroup_left = _get_subgroup_connectivity(conn_df, left_group)
                ft_left = conn_subgroup_left.to_numpy()
                
                # pdc is directed, no need to get symmetric matrix
                if ms_name != 'pdc':
                    # transform triangular matrix to symmetric square
                    ft_right = ft_right + ft_right.T - np.diag(np.diag(ft_right))
                    ft_left = ft_left + ft_left.T - np.diag(np.diag(ft_left))
                
                if ms_name == 'imcoh':
                    ft_right = abs(ft_right)
                    ft_left = abs(ft_left)
                
                # efficiencies
                right_efficiency = brainconn.distance.efficiency_wei(ft_right)
                left_efficiency = brainconn.distance.efficiency_wei(ft_left)
                # efficiency ratio
                efficiency_left_vs_right = _compute_ratio(left_efficiency,right_efficiency)
                # dict 
                eff_dict = {'efficiency_right': right_efficiency,
                            'efficiency_left': left_efficiency,
                            'efficiency_left_vs_right': efficiency_left_vs_right} 
                                
                # betweenness centralities
                bt_right = brainconn.centrality.betweenness_wei(ft_right)
                bt_left = brainconn.centrality.betweenness_wei(ft_left)
                # Mean, Std, Median, Ratios
                bt_dict = _compute_stats(bt_left, bt_right, 'betweness_centr')

                # add to dict                
                gr_ms_bds[bd_name] = eff_dict
                gr_ms_bds[bd_name].update(bt_dict)
                
                # pdc is directed -> different functions
                if ms_name == 'pdc':
                    # clustering cofficients
                    cc_right = brainconn.clustering.clustering_coef_wd(ft_right)
                    cc_left = brainconn.clustering.clustering_coef_wd(ft_left)
                    # Mean, Std, Median, Ratios
                    cc_dict = _compute_stats(cc_left, cc_right, 'clustering_coef')
                    
                    # strengths
                    in_right, out_right, _ = brainconn.degree.strengths_dir(ft_right)
                    in_left, out_left, _ = brainconn.degree.strengths_dir(ft_left)
                    
                    in_dict = _compute_stats(in_left, in_right, 'incoming_flow')
                    out_dict = _compute_stats(out_left, out_right, 'outgoing_flow')
                    
                    # save in dict
                    gr_ms_bds[bd_name].update(cc_dict)
                    gr_ms_bds[bd_name].update(in_dict)
                    gr_ms_bds[bd_name].update(out_dict)

                else:
                    # clustering cofficients
                    cc_right = brainconn.clustering.clustering_coef_wu(ft_right)
                    cc_left = brainconn.clustering.clustering_coef_wu(ft_left)
                    # Mean, Std, Median, Ratios
                    cc_dict = _compute_stats(cc_left, cc_right, 'clustering_coef')

                    st_right = brainconn.degree.strengths_und(ft_right)
                    st_left = brainconn.degree.strengths_und(ft_left)
                    
                    st_dict = _compute_stats(st_left, st_right, 'node_strengths')

                    # save in dict
                    gr_ms_bds[bd_name].update(cc_dict)
                    gr_ms_bds[bd_name].update(st_dict)
                
            gr_ms_measure[ms_name] = gr_ms_bds
        asymmetry_ms[filename] = gr_ms_measure
    
    return asymmetry_ms

#%% Run

_, fts = get_saved_features(withGraphs=False)
asymmetry_ms = compute_asymmetric_efficiencies(fts)
# createPickleFile(asymmetry_ms, '../2_Features_Data/128Hz/' + 'asymmetryMeasures')
