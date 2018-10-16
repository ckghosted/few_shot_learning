import os
import numpy as np
import pickle
import argparse

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
    parser.add_argument('--data_path', type=str, help='Path to save the produced train_novel, train_base, and test_novel, and test_base')
    args = parser.parse_args()
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)
    split_base_novel(args)

# Create sets of base classes (of size 80) and novel classes (of size 20, one for each superclass)
def split_base_novel(args):
	## Load cifar-100 raw data
    train_dict = unpickle(os.path.join(args.raw_path, 'train'))
    test_dict = unpickle(os.path.join(args.raw_path, 'test'))

    ## Make the {Superclass: {Classes}} dictionary
    class_mapping = {}
    for idx in range(len(train_dict[b'coarse_labels'])):
        if class_mapping.get(train_dict[b'coarse_labels'][idx]):
            class_mapping[train_dict[b'coarse_labels'][idx]].add(train_dict[b'fine_labels'][idx])
        else:
            class_mapping[train_dict[b'coarse_labels'][idx]] = set([train_dict[b'fine_labels'][idx]])
    for idx in range(len(test_dict[b'coarse_labels'])):
        if class_mapping.get(test_dict[b'coarse_labels'][idx]):
            class_mapping[test_dict[b'coarse_labels'][idx]].add(test_dict[b'fine_labels'][idx])
        else:
            class_mapping[test_dict[b'coarse_labels'][idx]] = set([test_dict[b'fine_labels'][idx]])
    #print(class_mapping)

    ## Randomly sample one class for each superclass to form the set of novel classes
    novel_classes = []
    for coarse_labels_idx in range(len(set(train_dict[b'coarse_labels']))):
        novel_classes.append(np.random.choice(list(class_mapping.get(coarse_labels_idx)), 1)[0])
    #print(novel_classes)

    ## Make train_novel_dict and train_base_dict
    novel_indexes = [idx for idx in range(len(train_dict[b'fine_labels'])) if train_dict[b'fine_labels'][idx] in novel_classes]
    train_novel_dict = {}
    train_novel_dict[b'coarse_labels'] = [train_dict[b'coarse_labels'][idx] for idx in novel_indexes]
    train_novel_dict[b'fine_labels'] = [train_dict[b'fine_labels'][idx] for idx in novel_indexes]
    train_novel_dict[b'data'] = train_dict[b'data'][novel_indexes]
    base_indexes = [idx for idx in range(len(train_dict[b'fine_labels'])) if train_dict[b'fine_labels'][idx] not in novel_classes]
    train_base_dict = {}
    train_base_dict[b'coarse_labels'] = [train_dict[b'coarse_labels'][idx] for idx in base_indexes]
    train_base_dict[b'fine_labels'] = [train_dict[b'fine_labels'][idx] for idx in base_indexes]
    train_base_dict[b'data'] = train_dict[b'data'][base_indexes]
    
    ## Make test_novel_dict and test_base_dict
    novel_indexes = [idx for idx in range(len(test_dict[b'fine_labels'])) if test_dict[b'fine_labels'][idx] in novel_classes]
    test_novel_dict = {}
    test_novel_dict[b'coarse_labels'] = [test_dict[b'coarse_labels'][idx] for idx in novel_indexes]
    test_novel_dict[b'fine_labels'] = [test_dict[b'fine_labels'][idx] for idx in novel_indexes]
    test_novel_dict[b'data'] = test_dict[b'data'][novel_indexes]
    base_indexes = [idx for idx in range(len(test_dict[b'fine_labels'])) if test_dict[b'fine_labels'][idx] not in novel_classes]
    test_base_dict = {}
    test_base_dict[b'coarse_labels'] = [test_dict[b'coarse_labels'][idx] for idx in base_indexes]
    test_base_dict[b'fine_labels'] = [test_dict[b'fine_labels'][idx] for idx in base_indexes]
    test_base_dict[b'data'] = test_dict[b'data'][base_indexes]

    ## Save
    dopickle(class_mapping, os.path.join(args.data_path, 'class_mapping'))
    dopickle(train_novel_dict, os.path.join(args.data_path, 'train_novel'))
    dopickle(train_base_dict, os.path.join(args.data_path, 'train_base'))
    dopickle(test_novel_dict, os.path.join(args.data_path, 'test_novel'))
    dopickle(test_base_dict, os.path.join(args.data_path, 'test_base'))

if __name__ == '__main__':
    main()