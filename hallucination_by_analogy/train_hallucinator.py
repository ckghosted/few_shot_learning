import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
from model_HAL import HAL
import os, re, glob

import argparse

import warnings
warnings.simplefilter('ignore')

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
    parser.add_argument('--quadruplet_name', type=str, help='File name of the saved quadruplet data (under the extractor folder)')
    parser.add_argument('--hallucinator_name', type=str, help='Folder name to save hallucinator models and learning curves')
    parser.add_argument('--loss_lambda', default=10.0, type=float, help='Scale to control the weighting of the MSE loss and the classification loss')
    parser.add_argument('--n_base_classes', default=80, type=int, help='Number of the base classes')
    parser.add_argument('--bsize', default=32, type=int, help='Batch size')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--l2scale', default=1e-3, type=float, help='L2-regularizer scale')
    parser.add_argument('--num_epoch', default=500, type=int, help='Max number of training epochs')
    parser.add_argument('--fc_dim', default=512, type=int, help='Fully-connected dimensions of the hidden layers of the MLP classifier')
    parser.add_argument('--patience', default=10, type=int, help='Patience for early-stopping mechanism')
    parser.add_argument('--debug', default=False, type=bool, help='Debug mode or not')
    args = parser.parse_args()
    train(args)

# Train the hallucinator
def train(args):
    train_for_hal_path = os.path.join(args.result_path, args.extractor_name, args.quadruplet_name)
    train_dict_for_hal = unpickle(train_for_hal_path)
    train_base_path = os.path.join(args.result_path, args.extractor_name, 'train_base_feat')
    train_base_dict = unpickle(train_base_path)
    if args.debug:
        print('Number of classes for training hallucinator: %d' % len(set(train_dict_for_hal[b'fine_labels'])))
        print('Number of base classes: %d' % len(set(train_base_dict[b'fine_labels'])))
        labels_in_train_for_hal = set(train_dict_for_hal[b'fine_labels'])
        labels_in_train_base = set(train_base_dict[b'fine_labels'])
        print('Base class(es) not appear in train_dict_for_hal: %s' % labels_in_train_base.difference(labels_in_train_for_hal))
    
    tf.reset_default_graph()
    with tf.Session() as sess:
        net = HAL(sess,
                  model_name=args.hallucinator_name,
                  result_path=args.result_path,
                  fc_dim=args.fc_dim,
                  n_fine_class=args.n_base_classes,
                  loss_lambda=args.loss_lambda)
        all_vars, trainable_vars, all_regs = net.build_model()
        res = net.train(train_path=train_for_hal_path,
                        train_base_path=train_base_path,
                        init_from=os.path.join(args.result_path, args.extractor_name, 'models'),
                        bsize=args.bsize,
                        learning_rate=args.learning_rate,
                        num_epoch=args.num_epoch,
                        patience=args.patience)
    np.save(os.path.join(args.result_path, args.hallucinator_name, 'results.npy'), res)
    
    # Debug: Check trainable variables and regularizers
    if args.debug:
        print('------------------[all_vars]------------------')
        for var in all_vars:
            print(var.name)
        print('------------------[trainable_vars]------------------')
        for var in trainable_vars:
            print(var.name)
        print('------------------[all_regs]------------------')
        for var in all_regs:
            print(var.name)
    
    # Plot learning curve
    results = np.load(os.path.join(args.result_path, args.hallucinator_name, 'results.npy'))
    fig, ax = plt.subplots(1,1, figsize=(8,6))
    ax.plot(range(1, len(results[0])+1), results[0], label='Training error')
    ax.plot(range(1, len(results[1])+1), results[1], label='Validation error')
    ax.set_xticks(np.arange(1, len(results[0])+1))
    ax.set_xlabel('Training epochs', fontsize=16)
    ax.set_ylabel('%4f * MSE + classification error' % args.loss_lambda, fontsize=16)
    ax.legend(fontsize=16)
    plt.suptitle('Learning Curve', fontsize=20)
    fig.savefig(os.path.join(args.result_path, args.hallucinator_name, 'learning_curve.jpg'),
                bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    main()