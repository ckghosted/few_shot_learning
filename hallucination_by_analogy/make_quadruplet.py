import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import os, re, glob

import argparse

import warnings
warnings.simplefilter('ignore')

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scipy import spatial
import tqdm

import pickle

def dopickle(dict_, file):
    with open(file, 'wb') as fo:
        pickle.dump(dict_, fo)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, help='Path to save all results')
    parser.add_argument('--extractor_name', type=str, help='Folder name of the saved extractor model')
    parser.add_argument('--quadruplet_name', type=str, help='File name to save the quadruplet data (under the extractor folder)')
    parser.add_argument('--n_clusters', default=10, type=int, help='Number of clusters used in the KMeans clustering algorithm')
    parser.add_argument('--fc_dim', default=512, type=int, help='Fully-connected dimensions of the hidden layers of the MLP classifier')
    parser.add_argument('--debug', default=False, type=bool, help='Debug mode or not')
    args = parser.parse_args()
    make_quadruplet(args)

# Make training data for the hallucinator
def make_quadruplet(args):
    train_base_path = os.path.join(args.result_path, args.extractor_name, 'train_base_feat')
    train_base_dict = unpickle(train_base_path)
    
    ## (1) For each class, make args.n_cluster clusters
    ### all_possible_combination: an (args.n_cluster)-by-2 array containing all possible index combinations:
    ### [[0 0], [0 1], [0 2], ..., [1 0], [1 1], [1 2], ...]
    all_possible_combination = np.transpose([np.repeat(np.arange(args.n_clusters), args.n_clusters),
                                             np.tile(np.arange(args.n_clusters), args.n_clusters)])
    feature_pair_collection = {}
    for lb in set(train_base_dict[b'fine_labels']):
        if args.debug:
            print('fine_label: %d' % lb)
        indexes_for_this_lb = [idx for idx in range(len(train_base_dict[b'fine_labels'])) \
                               if train_base_dict[b'fine_labels'][idx] == lb]
        feat_for_this_lb = train_base_dict[b'features'][indexes_for_this_lb]
        kmeans = KMeans(n_clusters=args.n_clusters).fit(feat_for_this_lb)
        if args.debug:
            print(kmeans.labels_)
        feature_pair_collection[lb] = kmeans.cluster_centers_[all_possible_combination]
    
    ## (2) For each pair of class, collect "similar" pairs of centers
    quadruplet_collection = None
    fine_label_collection = []
    for lb1 in set(train_base_dict[b'fine_labels']):
        if args.debug:
            print('class %d' % lb1)
        for idx1 in tqdm.tqdm(range(feature_pair_collection[lb1].shape[0])):
            max_cos_sim = 0
            best_lb2 = 0
            best_idx2 = 0
            for lb2 in set(train_base_dict[b'fine_labels']):
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
                fine_label_collection.append(best_lb2)
        if args.debug and quadruplet_collection is not None:
            print('    quadruplet_collection.shape: %s, len(fine_label_collection) = %d' % \
                  (quadruplet_collection.shape, len(fine_label_collection)))
    
    ## (3) Re-arrange and save
    train_dict_for_hal = {}
    train_dict_for_hal[b'fine_labels'] = fine_label_collection
    if args.debug:
        print(len(train_dict_for_hal[b'fine_labels']))
    fc_dim = quadruplet_collection.shape[-1]
    train_dict_for_hal[b'triplet_features'] = quadruplet_collection[:,0:3,:].reshape((-1, fc_dim*3))
    if args.debug:
        print(train_dict_for_hal[b'triplet_features'].shape)
    train_dict_for_hal[b'target_features'] = quadruplet_collection[:,3,:].reshape((-1, fc_dim))
    if args.debug:
        print(train_dict_for_hal[b'target_features'].shape)
    train_for_hal_path = os.path.join(args.result_path, args.extractor_name, args.quadruplet_name)
    dopickle(train_dict_for_hal, train_for_hal_path)

if __name__ == '__main__':
    main()