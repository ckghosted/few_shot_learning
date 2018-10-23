import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import os, re, glob, time

import argparse

import warnings
warnings.simplefilter('ignore')

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scipy import spatial

import pickle

def dopickle(dict_, file):
    with open(file, 'wb') as fo:
        pickle.dump(dict_, fo)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_

import multiprocessing as mp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path of the saved all_base_labels list and class_mapping dictionary')
    parser.add_argument('--result_path', type=str, help='Path to save all results')
    parser.add_argument('--extractor_name', type=str, help='Folder name of the saved extractor model')
    parser.add_argument('--quadruplet_name', type=str, help='File name to save the quadruplet data (under the extractor folder)')
    parser.add_argument('--n_clusters', default=10, type=int, help='Number of clusters used in the KMeans clustering algorithm')
    parser.add_argument('--fc_dim', default=512, type=int, help='Fully-connected dimensions of the hidden layers of the MLP classifier')
    parser.add_argument('--n_cores', default=8, type=int, help='Number of CPU cores used to execute this program')
    parser.add_argument('--debug', default=False, type=bool, help='Debug mode or not')
    args = parser.parse_args()
    feature_pair_collection = kmeans_algo(args)
    if args.debug:
        print('==================== kmeans_algo finished ====================')

    ## Reference: https://sebastianraschka.com/Articles/2014_multiprocessing.html
    output = mp.Queue()
    processes = [mp.Process(target=make_quadruplet, args=(args, feature_pair_collection, subset_idx, output)) \
                 for subset_idx in range(args.n_cores)]
    
    for p in processes:
        p.start()
    
    if args.debug:
        print('==================== all processes started ====================')
    
    results = {}
    results[b'ref_labels'] = []
    results[b'fine_labels'] = []
    results[b'cos_sim'] = []
    results[b'triplet_features'] = None
    results[b'target_features'] = None
    
    for p in processes:
        res = output.get()
        results[b'ref_labels'].extend(res[b'ref_labels'])
        results[b'fine_labels'].extend(res[b'fine_labels'])
        results[b'cos_sim'].extend(res[b'cos_sim'])
        if results[b'target_features'] is None:
            results[b'target_features'] = res[b'target_features']
        else:
            results[b'target_features'] = np.concatenate((results[b'target_features'], res[b'target_features']), axis=0)
        if results[b'triplet_features'] is None:
            results[b'triplet_features'] = res[b'triplet_features']
        else:
            results[b'triplet_features'] = np.concatenate((results[b'triplet_features'], res[b'triplet_features']), axis=0)
    
    if args.debug:
        print('==================== all results collected ====================')
    
    ## "Whenever you use a queue, you need to make sure that all items which have been put on the queue
    ##  will eventually be removed before the process is joined."
    ## Reference: https://stackoverflow.com/questions/44581371/python-multiprocessing-imported-functions-not-ending
    for p in processes:
        p.join()
    
    if args.debug:
        print('==================== all processes joined ====================')

    train_for_hal_path = os.path.join(args.result_path, args.extractor_name, args.quadruplet_name)
    dopickle(results, train_for_hal_path)

# For each class, make args.n_cluster clusters
def kmeans_algo(args):
    train_base_path = os.path.join(args.result_path, args.extractor_name, 'train_base_feat')
    train_base_dict = unpickle(train_base_path)
    
    ## all_index_combination: an (args.n_cluster)-by-2 array containing all possible index combinations:
    ## [[0 0], [0 1], [0 2], ..., [1 0], [1 1], [1 2], ...]
    all_index_combination = np.transpose([np.repeat(np.arange(args.n_clusters), args.n_clusters),
                                         np.tile(np.arange(args.n_clusters), args.n_clusters)])
    ## Remove index combinations that have the same index: [0 0], [1 1], ...
    all_possible_combination = all_index_combination[[(lst[0] != lst[1]) for lst in all_index_combination]]
    feature_pair_collection = {}
    for lb in set(train_base_dict[b'fine_labels']):
        if args.debug:
            print('fine_label: %d' % lb)
        indexes_for_this_lb = [idx for idx in range(len(train_base_dict[b'fine_labels'])) \
                               if train_base_dict[b'fine_labels'][idx] == lb]
        feat_for_this_lb = train_base_dict[b'features'][indexes_for_this_lb]
        kmeans = KMeans(n_clusters=args.n_clusters, n_jobs=8).fit(feat_for_this_lb)
        if args.debug:
            print(kmeans.labels_)
        feature_pair_collection[lb] = kmeans.cluster_centers_[all_possible_combination]
    return feature_pair_collection

# For each pair of classes, collect "similar" pairs of centers to make training quadruplets
def make_quadruplet(args, feature_pair_collection, subset_idx, output):
    all_base_labels = unpickle(os.path.join(args.data_path, 'all_base_labels'))
    lb_per_core = int(len(all_base_labels) / args.n_cores)
    if args.debug:
        print('subset_idx = %d, len(feature_pair_collection) = %d, feature_pair_collection[0].shape = %s' \
            % (subset_idx, len(feature_pair_collection), feature_pair_collection[0].shape))
        print('    selected_base_labels: ', end='')
        print(all_base_labels[subset_idx*lb_per_core:(subset_idx+1)*lb_per_core])
    
    ## Make all_possible_combination again (just like that in kmeans_algo())
    all_index_combination = np.transpose([np.repeat(np.arange(args.n_clusters), args.n_clusters),
                                         np.tile(np.arange(args.n_clusters), args.n_clusters)])
    all_possible_combination = all_index_combination[[(lst[0] != lst[1]) for lst in all_index_combination]]
    ## Actually we don't need to consider all possible combination of indexes:
    ## Suppose that we have checked centroid indexes [0 1] of a label (lb1), and found that
    ## centroid indexes [x y] of some other label (lb2) have the maximum cosine similarity.
    ## Then, if we check centroid indexes [1 0] of lb1, we must find centroid indexes [y x]
    ## of lb2 have the maximum cosine similarity.
    considered_idx1 = [idx for idx in range(all_possible_combination.shape[0]) \
                       if all_possible_combination[idx][0] < all_possible_combination[idx][1]]
    
    ## Make an inverse mapping from (base) fine labels to the corresponding coarse labels
    class_mapping = unpickle(os.path.join(args.data_path, 'class_mapping'))
    for coarse_lb in class_mapping:
        temp_set = set()
        for fine_lb in class_mapping[coarse_lb]:
            if fine_lb in all_base_labels:
                temp_set.add(fine_lb)
        class_mapping[coarse_lb] = temp_set
    class_mapping_inv = {}
    for fine_lb in all_base_labels:
        for coarse_lb in class_mapping.keys():
            if fine_lb in class_mapping[coarse_lb]:
                class_mapping_inv[fine_lb] = coarse_lb
                break
    if args.debug:
        print(class_mapping_inv)
    
    quadruplet_collection = None
    fine_label_collection1 = []
    fine_label_collection2 = []
    cos_sim_collection = []
    ## Instead of looping all possible base classes, use "subset_idx" to indicate a subset of base classes
    for lb1 in all_base_labels[subset_idx*lb_per_core:(subset_idx+1)*lb_per_core]:
        if args.debug:
            print('class %d' % lb1)
        for idx1 in considered_idx1:
            max_cos_sim = 0
            best_lb2 = 0
            best_idx2 = 0
            ### Consider only those fine labels belonging to the same coarse labels
            for lb2 in class_mapping[class_mapping_inv[lb1]]:
                if lb1 != lb2:
                    for idx2 in range(feature_pair_collection[lb2].shape[0]):
                        center_diff_1 = feature_pair_collection[lb1][idx1][0] - feature_pair_collection[lb1][idx1][1]
                        center_diff_2 = feature_pair_collection[lb2][idx2][0] - feature_pair_collection[lb2][idx2][1]
                        cos_sim = 1 - spatial.distance.cosine(center_diff_1, center_diff_2)
                        if cos_sim > max_cos_sim:
                            max_cos_sim = cos_sim
                            best_lb2 = lb2
                            best_idx2 = idx2
            if args.debug:
                print('    max_cos_sim: %.4f' % max_cos_sim)
            if max_cos_sim > 0: ## False if max_cos_sim is nan
                quadruplet = np.concatenate((np.expand_dims(feature_pair_collection[lb1][idx1][0], 0),
                                             np.expand_dims(feature_pair_collection[lb1][idx1][1], 0),
                                             np.expand_dims(feature_pair_collection[best_lb2][best_idx2][0], 0),
                                             np.expand_dims(feature_pair_collection[best_lb2][best_idx2][1], 0)), axis=0)
                if quadruplet_collection is None:
                    quadruplet_collection = np.expand_dims(quadruplet, 0)
                else:
                    quadruplet_collection = np.concatenate((quadruplet_collection, np.expand_dims(quadruplet, 0)), axis=0)
                fine_label_collection1.append(lb1) ## Also record lb1 for convenience
                fine_label_collection2.append(best_lb2)
                cos_sim_collection.append(max_cos_sim)
                ## For quadruplet (a1, a2, b1, b2), also add (a2, a1, b2, b1)
                quadruplet = np.concatenate((np.expand_dims(feature_pair_collection[lb1][idx1][1], 0),
                                             np.expand_dims(feature_pair_collection[lb1][idx1][0], 0),
                                             np.expand_dims(feature_pair_collection[best_lb2][best_idx2][1], 0),
                                             np.expand_dims(feature_pair_collection[best_lb2][best_idx2][0], 0)), axis=0)
                quadruplet_collection = np.concatenate((quadruplet_collection, np.expand_dims(quadruplet, 0)), axis=0)
                fine_label_collection1.append(lb1) ## Also record lb1 for convenience
                fine_label_collection2.append(best_lb2)
                cos_sim_collection.append(max_cos_sim)
        if args.debug and quadruplet_collection is not None:
            print('    quadruplet_collection.shape: %s, len(fine_label_collection2) = %d' % \
                  (quadruplet_collection.shape, len(fine_label_collection2)))
        print('label %d done' % lb1)
    
    ## Re-arrange and save
    train_dict_for_hal = {}
    train_dict_for_hal[b'ref_labels'] = fine_label_collection1
    train_dict_for_hal[b'fine_labels'] = fine_label_collection2
    train_dict_for_hal[b'cos_sim'] = cos_sim_collection
    if args.debug:
        print(len(train_dict_for_hal[b'ref_labels']))
    if args.debug:
        print(len(train_dict_for_hal[b'fine_labels']))
    train_dict_for_hal[b'triplet_features'] = quadruplet_collection[:,0:3,:].reshape((-1, args.fc_dim*3))
    if args.debug:
        print(train_dict_for_hal[b'triplet_features'].shape)
    train_dict_for_hal[b'target_features'] = quadruplet_collection[:,3,:].reshape((-1, args.fc_dim))
    if args.debug:
        print(train_dict_for_hal[b'target_features'].shape)
    
    output.put(train_dict_for_hal)
    if args.debug:
        print('[FINISHED] subset_idx = %d' % subset_idx)

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end - start)
