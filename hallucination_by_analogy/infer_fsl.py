import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
from model_FSL import FSL
import os, re, glob

import argparse

import warnings
warnings.simplefilter('ignore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, help='Path to save all results')
    parser.add_argument('--model_name', type=str, help='Folder name to save FSL models and learning curves')
    parser.add_argument('--extractor_name', type=str, help='Folder name of the saved extractor model')
    parser.add_argument('--hallucinator_name', type=str, help='Folder name of the saved hallucinator model')
    parser.add_argument('--n_fine_classes', default=100, type=int, help='Number of classes (base + novel)')
    parser.add_argument('--n_top', default=5, type=int, help='Number to compute the top-n accuracy')
    parser.add_argument('--bsize', default=64, type=int, help='Batch size')
    parser.add_argument('--fc_dim', default=512, type=int, help='Fully-connected dimensions of the hidden layers of the MLP classifier')
    parser.add_argument('--debug', default=False, type=bool, help='Debug mode or not')
    args = parser.parse_args()
    inference(args)

# Inference
def inference(args):
    print('============================ inference ============================')
    tf.reset_default_graph()
    with tf.Session() as sess:
        net = FSL(sess,
                  model_name=args.model_name,
                  result_path=os.path.join(args.result_path, args.hallucinator_name),
                  fc_dim=args.fc_dim,
                  n_fine_class=args.n_fine_classes)
        net.build_model()
        net.inference(test_novel_path=os.path.join(args.result_path, args.extractor_name, 'test_novel_feat'),
                      test_base_path=os.path.join(args.result_path, args.extractor_name, 'test_base_feat'),
                      gen_from=os.path.join(args.result_path, args.hallucinator_name, args.model_name, 'models'),
                      out_path=os.path.join(args.result_path, args.hallucinator_name, args.model_name),
                      n_top=args.n_top,
                      bsize=args.bsize)

if __name__ == '__main__':
    main()