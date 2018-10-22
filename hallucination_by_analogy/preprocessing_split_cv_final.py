import os
import numpy as np
import pickle
import argparse
import shutil

def dopickle(dict_, file):
    with open(file, 'wb') as fo:
        pickle.dump(dict_, fo)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', type=str, help='Path of cifar-100 raw data (train and test)')
    parser.add_argument('--data_path', type=str, help='Path to save the produced cv/train, cv/test, final/train, and final/test')
    args = parser.parse_args()
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)
        os.makedirs(os.path.join(args.data_path, 'cv'))
        os.makedirs(os.path.join(args.data_path, 'final'))
    split_cv_final(args)

# Create two sets of superclasses for cross-validation and final evaluation, respectively, each of size 10
def split_cv_final(args):
    ## Load cifar-100 raw data
    train_dict = unpickle(os.path.join(args.raw_path, 'train'))
    test_dict = unpickle(os.path.join(args.raw_path, 'test'))
    
    cv_coarse_labels = list(np.random.choice(list(set(train_dict[b'coarse_labels'])), 10, replace=False))
    print('cv_coarse_labels: ', end='')
    print(cv_coarse_labels)
    
    cv_indexes = [idx for idx in range(len(train_dict[b'coarse_labels'])) if train_dict[b'coarse_labels'][idx] in cv_coarse_labels]
    final_indexes = [idx for idx in range(len(train_dict[b'coarse_labels'])) if not train_dict[b'coarse_labels'][idx] in cv_coarse_labels]
    train_cv_dict = {}
    train_cv_dict[b'coarse_labels'] = [train_dict[b'coarse_labels'][idx] for idx in cv_indexes]
    train_cv_dict[b'fine_labels'] = [train_dict[b'fine_labels'][idx] for idx in cv_indexes]
    train_cv_dict[b'data'] = train_dict[b'data'][cv_indexes]
    train_final_dict = {}
    train_final_dict[b'coarse_labels'] = [train_dict[b'coarse_labels'][idx] for idx in final_indexes]
    train_final_dict[b'fine_labels'] = [train_dict[b'fine_labels'][idx] for idx in final_indexes]
    train_final_dict[b'data'] = train_dict[b'data'][final_indexes]
    
    cv_indexes = [idx for idx in range(len(test_dict[b'coarse_labels'])) if test_dict[b'coarse_labels'][idx] in cv_coarse_labels]
    final_indexes = [idx for idx in range(len(test_dict[b'coarse_labels'])) if not test_dict[b'coarse_labels'][idx] in cv_coarse_labels]
    test_cv_dict = {}
    test_cv_dict[b'coarse_labels'] = [test_dict[b'coarse_labels'][idx] for idx in cv_indexes]
    test_cv_dict[b'fine_labels'] = [test_dict[b'fine_labels'][idx] for idx in cv_indexes]
    test_cv_dict[b'data'] = test_dict[b'data'][cv_indexes]
    test_final_dict = {}
    test_final_dict[b'coarse_labels'] = [test_dict[b'coarse_labels'][idx] for idx in final_indexes]
    test_final_dict[b'fine_labels'] = [test_dict[b'fine_labels'][idx] for idx in final_indexes]
    test_final_dict[b'data'] = test_dict[b'data'][final_indexes]
    
    ## Save
    dopickle(cv_coarse_labels, os.path.join(args.data_path, 'cv_coarse_labels'))
    dopickle(train_cv_dict, os.path.join(args.data_path, 'cv', 'train'))
    dopickle(test_cv_dict, os.path.join(args.data_path, 'cv', 'test'))
    dopickle(train_final_dict, os.path.join(args.data_path, 'final', 'train'))
    dopickle(test_final_dict, os.path.join(args.data_path, 'final', 'test'))
    
    ## Copy 'meta'
    shutil.copy(os.path.join(args.raw_path, 'meta'), os.path.join(args.data_path, 'cv', 'meta'))
    shutil.copy(os.path.join(args.raw_path, 'meta'), os.path.join(args.data_path, 'final', 'meta'))

if __name__ == '__main__':
    main()