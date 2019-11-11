import os, sys
import csv
import numpy as np 
from numpy import genfromtxt

def subsample(adj, features):
    # input: two sparse matrices

    nb_nodes = adj.shape[0]
    one_hop_features = adj @ features

    features_list = []
    for i in range(nb_nodes):
        features_list.append([])
    idxs0, idxs1 = features.nonzero()
    for i in range(idxs0.shape[0]):
        features_list[idxs0[i]].append(idxs1[i])

    one_hop_features_list = []
    for i in range(nb_nodes):
        one_hop_features_list.append([])
    idxs0, idxs1 = one_hop_features.nonzero()
    for i in range(idxs0.shape[0]):
        one_hop_features_list[idxs0[i]].append(idxs1[i])
    
    
    subsample_features_list = []
    for i in range(nb_nodes):
        subsample_features_list.append(list(set(features_list[i]) & set(one_hop_features_list[i])))
    '''

    subsample_features_list = []
    for i in range(nb_nodes):
        subsample_features_list.append(list(set(features_list[i]).difference(set(one_hop_features_list[i]))))
    '''
    
    for i in range(nb_nodes):
        for ft in subsample_features_list[i]:
            features[i, ft] = 0
    
    features.eliminate_zeros()
    
    return features

def output_csv(file_name, score_list):
    with open(file_name, mode='w') as output_file:
        output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for score in score_list:
            output_writer.writerow([score])

def average_csv(dir, prefix):
    file_list = os.listdir(dir)
    
    i = 0
    for file in file_list:
        if file.startswith(prefix):
            if i == 0:
                res = np.expand_dims(genfromtxt(dir + '/' + file, delimiter=','), axis=0)
            else:
                res = np.append(res, np.expand_dims(genfromtxt(dir + '/' + file, delimiter=','), axis=0), axis=0)
            i += 1
    print(res.shape)
    print(np.mean(res, axis=0))
    print(np.std(res, axis=0))