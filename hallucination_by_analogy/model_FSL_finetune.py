import os, re, time, glob
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer

from sklearn.metrics import accuracy_score

from ops import *
from utils import *

import pickle
import tqdm

import imgaug as ia
from imgaug import augmenters as iaa

VGG_MEAN = [103.939, 116.779, 123.68]
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

def dopickle(dict_, file):
    with open(file, 'wb') as fo:
        pickle.dump(dict_, fo)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_

from model_VGG import VGG

# FSL that takes raw images as inputs and uses a pre-trained CNN-based feature extractor
# to extract features from raw images.
# (Allow fine-tuning the CNN-based feature extractor, not implemented yet.)
class VGG_FSL_FT(VGG):
    def __init__(self,
                 sess,
                 model_name='VGG_FSL_FT',
                 result_path='/home/cclin/few_shot_learning/hallucination_by_analogy/results',
                 img_size_h=32,
                 img_size_w=32,
                 c_dim=3,
                 fc_dim=512,
                 n_fine_class=100,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 l2scale=0.001,
                 vgg16_npy_path='/data/put_data/cclin/ntu/dlcv2018/hw3/vgg16.npy'):
        super(VGG_FSL_FT, self).__init__(sess,
                                         model_name,
                                         result_path,
                                         img_size_h,
                                         img_size_w,
                                         c_dim,
                                         fc_dim,
                                         n_fine_class,
                                         bnDecay,
                                         epsilon,
                                         l2scale,
                                         vgg16_npy_path)
    
    def train(self,
              train_novel_path='/data/put_data/cclin/datasets/cifar-100-python/train_novel',
              train_base_path='/data/put_data/cclin/datasets/cifar-100-python/train_base',
              init_from=None, ## e.g., model_name (if None ==> train from scratch)
              n_shot=1,
              n_top=5, ## top-n accuracy
              bsize=32,
              learning_rate=5e-5,
              num_epoch=50,
              patience=10,
              ratio_for_dense=0.0): ## ratio_for_dense: Not used anymore
        ### create a dedicated folder for this model
        if os.path.exists(os.path.join(self.result_path, self.model_name)):
            print('WARNING: the folder "{}" already exists!'.format(os.path.join(self.result_path, self.model_name)))
        else:
            os.makedirs(os.path.join(self.result_path, self.model_name))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'models'))
        
        ### Load training data (as two dictionaries) from both base and novel classes,
        ### and split each of them into training/validation by 80/20
        train_novel_dict = unpickle(train_novel_path)
        data_len = len(train_novel_dict[b'fine_labels'])
        data_novel_train = train_novel_dict[b'data'][0:int(data_len*0.8)].reshape((-1, 3, 32, 32)).transpose([0, 2, 3, 1])
        data_novel_valid = train_novel_dict[b'data'][int(data_len*0.8):int(data_len)].reshape((-1, 3, 32, 32)).transpose([0, 2, 3, 1])
        fine_labels = [int(s) for s in train_novel_dict[b'fine_labels'][0:int(data_len*0.8)]]
        labels_novel_train = np.eye(self.n_fine_class)[fine_labels]
        fine_labels = [int(s) for s in train_novel_dict[b'fine_labels'][int(data_len*0.8):int(data_len)]]
        labels_novel_valid = np.eye(self.n_fine_class)[fine_labels]
        train_base_dict = unpickle(train_base_path)
        data_len = len(train_base_dict[b'fine_labels'])
        data_base_train = train_base_dict[b'data'][0:int(data_len*0.8)].reshape((-1, 3, 32, 32)).transpose([0, 2, 3, 1])
        data_base_valid = train_base_dict[b'data'][int(data_len*0.8):int(data_len)].reshape((-1, 3, 32, 32)).transpose([0, 2, 3, 1])
        fine_labels = [int(s) for s in train_base_dict[b'fine_labels'][0:int(data_len*0.8)]]
        labels_base_train = np.eye(self.n_fine_class)[fine_labels]
        fine_labels = [int(s) for s in train_base_dict[b'fine_labels'][int(data_len*0.8):int(data_len)]]
        labels_base_valid = np.eye(self.n_fine_class)[fine_labels]
        
        ### For the training split, use all base samples and randomly selected novel samples
        selected_indexes = []
        for lb in set(np.argmax(labels_novel_train, axis=1)):
            candidate_indexes_per_lb = [idx for idx in range(labels_novel_train.shape[0]) if np.argmax(labels_novel_train[idx]) == lb]
            selected_indexes_per_lb = np.random.choice(candidate_indexes_per_lb, n_shot)
            selected_indexes.extend(selected_indexes_per_lb)
        data_novel_train_selected = data_novel_train[selected_indexes]
        labels_novel_train_selected = labels_novel_train[selected_indexes]
        data_train = np.concatenate((data_novel_train_selected, data_base_train), axis=0)
        fine_labels_train = np.concatenate((labels_novel_train_selected, labels_base_train), axis=0)
        nBatches = int(np.ceil(data_train.shape[0] / bsize))
        
        ### For the validation split, just combine all base samples and all novel samples
        data_valid = np.concatenate((data_novel_valid, data_base_valid), axis=0)
        fine_labels_valid = np.concatenate((labels_novel_valid, labels_base_valid), axis=0)
        nBatches_valid = int(np.ceil(data_valid.shape[0] / bsize))
        
        ### Data indexes used to shuffle training order
        arr = np.arange(data_train.shape[0])
        
        ## Basic image augmentation
        seq = iaa.Sequential([
            iaa.Crop(px=(0, 4)), # crop images from each side by 0 to 4px (randomly chosen)
            iaa.Fliplr(0.5), # horizontally flip 50% of the images
            iaa.GaussianBlur(sigma=(0, 0.5)), # blur images with a sigma of 0 to 0.5
            sometimes(iaa.Affine(
                scale={"x": (0.85, 1.15), "y": (0.85, 1.15)}, # scale images to 85-115% of their size, individually per axis
                translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}, # translate by -15 to +15 percent (per axis)
                rotate=(-30, 30), # rotate by -30 to +30 degrees
                shear=(-12, 12), # shear by -12 to +12 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            ))
        ])
        
        ## initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)
        
        ## load previously-trained feature extractor
        if init_from is not None:
            could_load, checkpoint_counter = self.load_cnn(init_from)
            if could_load:
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        else:
            print(" [@] train from scratch")
        
        ## main training loop
        loss_train = []
        loss_valid = []
        acc_train = []
        acc_valid = []
        top_n_acc_train = []
        top_n_acc_valid = []
        best_loss = 0
        stopping_step = 0
        for epoch in range(1, (num_epoch+1)):
            loss_train_batch = []
            loss_valid_batch = []
            acc_train_batch = []
            acc_valid_batch = []
            top_n_acc_train_batch = []
            top_n_acc_valid_batch = []
            ### shuffle training order for each epoch
            np.random.shuffle(arr)
            #print('training')
            for idx in tqdm.tqdm(range(nBatches)):
                batch_data = data_train[arr[idx*bsize:(idx+1)*bsize]]
                batch_data_aug = seq.augment_images(batch_data)  # done by the library
                batch_labels = fine_labels_train[arr[idx*bsize:(idx+1)*bsize]]
                #print(batch_labels.shape)
                if could_load: ## train dense layers (i.e., the classifier) only
                    _, loss, logits = self.sess.run([self.opt_mlp, self.loss, self.dense16],
                                                    feed_dict={self.images: batch_data_aug,
                                                               self.fine_labels: batch_labels,
                                                               self.bn_train: True,
                                                               self.keep_prob: 0.5,
                                                               self.learning_rate: learning_rate})
                else:
                    _, loss, logits = self.sess.run([self.opt_all, self.loss, self.dense16],
                                                    feed_dict={self.images: batch_data_aug,
                                                               self.fine_labels: batch_labels,
                                                               self.bn_train: True,
                                                               self.keep_prob: 0.5,
                                                               self.learning_rate: learning_rate})
                loss_train_batch.append(loss)
                y_true = np.argmax(batch_labels, axis=1)
                y_pred = np.argmax(logits, axis=1)
                acc_train_batch.append(accuracy_score(y_true, y_pred))
                best_n = np.argsort(logits, axis=1)[:,-n_top:]
                top_n_acc_train_batch.append(np.mean([(y_true[batch_idx] in best_n[batch_idx]) for batch_idx in range(len(y_true))]))
            ### compute validation loss
            #print('validation')
            for idx in tqdm.tqdm(range(nBatches_valid)):
                batch_data = data_valid[idx*bsize:(idx+1)*bsize]
                batch_labels = fine_labels_valid[idx*bsize:(idx+1)*bsize]
                #print(batch_labels.shape)
                loss, logits = self.sess.run([self.loss, self.dense16],
                                             feed_dict={self.images: batch_data,
                                                        self.fine_labels: batch_labels,
                                                        self.bn_train: False,
                                                        self.keep_prob: 1.0,})
                loss_valid_batch.append(loss)
                y_true = np.argmax(batch_labels, axis=1)
                y_pred = np.argmax(logits, axis=1)
                acc_valid_batch.append(accuracy_score(y_true, y_pred))
                best_n = np.argsort(logits, axis=1)[:,-n_top:]
                top_n_acc_valid_batch.append(np.mean([(y_true[batch_idx] in best_n[batch_idx]) for batch_idx in range(len(y_true))]))
            ### record training loss for each epoch (instead of each iteration)
            loss_train.append(np.mean(loss_train_batch))
            loss_valid.append(np.mean(loss_valid_batch))
            acc_train.append(np.mean(acc_train_batch))
            acc_valid.append(np.mean(acc_valid_batch))
            top_n_acc_train.append(np.mean(top_n_acc_train_batch))
            top_n_acc_valid.append(np.mean(top_n_acc_valid_batch))
            print('Epoch: %d, train loss: %f, valid loss: %f, train accuracy: %f, valid accuracy: %f' % \
                  (epoch, np.mean(loss_train_batch), np.mean(loss_valid_batch), np.mean(acc_train_batch), np.mean(acc_valid_batch)))
            print('           top %d train accuracy: %f, top %d valid accuracy: %f' % \
                  (n_top, np.mean(top_n_acc_train_batch), n_top, np.mean(top_n_acc_valid_batch)))
            
            ### save model if improvement, stop if reach patience
            current_loss = np.mean(loss_valid_batch)
            current_acc = np.mean(acc_valid_batch)
            if epoch == 1:
                best_loss = current_loss
                best_acc = current_acc
            else:
                #if current_loss < best_loss or current_acc > best_acc:
                if current_loss < best_loss: ## only monitor loss
                    best_loss = current_loss
                    best_acc = current_acc
                    self.saver.save(self.sess,
                                    os.path.join(self.result_path, self.model_name, 'models', self.model_name + '.model'),
                                    global_step=epoch)
                    self.saver_cnn.save(self.sess,
                                        os.path.join(self.result_path, self.model_name, 'models_cnn', self.model_name + '.model-cnn'),
                                        global_step=epoch)
                    stopping_step = 0
                else:
                    stopping_step += 1
                print('stopping_step = %d' % stopping_step)
                if stopping_step >= patience:
                    print('stopping_step >= patience (%d), stop training' % patience)
                    break
        return [loss_train, loss_valid, acc_train, acc_valid]
    
    def inference(self,
                  #test_path='/data/put_data/cclin/datasets/cifar-100-python/test_novel', ### for close-world evaluation
                  test_path='/data/put_data/cclin/datasets/cifar-100-python/test', ### for open-world evaluation
                  gen_from=None, ## e.g., model_name (must given)
                  gen_from_ckpt=None, ## e.g., model_name+'.model-1680' (can be None)
                  out_path=None,
                  n_top=5, ## top-n accuracy
                  bsize=32):
        ## create output folder
        if gen_from is None:
            gen_from = os.path.join(self.result_path, self.model_name, 'models')
        if out_path is not None:
            if os.path.exists(out_path):
                print('WARNING: the output path "{}" already exists!'.format(out_path))
            else:
                os.makedirs(out_path)
        else:
            out_path = os.path.join(self.result_path, self.model_name)
        
        ## load previous model if possible
        could_load, checkpoint_counter = self.load(gen_from, gen_from_ckpt)
        if could_load:
            print(" [*] Load SUCCESS")
            
            #### load testing data
            test_dict = unpickle(test_path)
            ### [3072] with the first 1024 being 'R', the middle 1024 being 'G', and the last 1024 being 'B'
            ### reshape and modify rank order from (batch, channel, height, width) to (batch, height, width, channel)
            data_test = test_dict[b'data'].reshape((-1, 3, 32, 32)).transpose([0, 2, 3, 1])
            nBatches_test = int(np.ceil(data_test.shape[0] / bsize))
            #### one-hot
            fine_labels = [int(s) for s in test_dict[b'fine_labels']]
            fine_labels_test = np.eye(self.n_fine_class)[fine_labels]
            
            ### make prediction and compute accuracy
            loss_test_batch=[]
            acc_test_batch=[]
            top_n_acc_test_batch = []
            for idx in tqdm.tqdm(range(nBatches_test)):
                batch_data = data_test[idx*bsize:(idx+1)*bsize]
                batch_labels = fine_labels_test[idx*bsize:(idx+1)*bsize]
                loss, logits = self.sess.run([self.loss, self.dense16],
                                             feed_dict={self.images: batch_data,
                                                        self.fine_labels: batch_labels,
                                                        self.bn_train: False,
                                                        self.keep_prob: 1.0,})
                loss_test_batch.append(loss)
                y_true = np.argmax(batch_labels, axis=1)
                y_pred = np.argmax(logits, axis=1)
                acc_test_batch.append(accuracy_score(y_true, y_pred))
                best_n = np.argsort(logits, axis=1)[:,-n_top:]
                top_n_acc_test_batch.append(np.mean([(y_true[batch_idx] in best_n[batch_idx]) for batch_idx in range(len(y_true))]))
            print('test loss: %f, test accuracy: %f, top-%d test accuracy: %f' % \
                  (np.mean(loss_test_batch), np.mean(acc_test_batch), n_top, np.mean(top_n_acc_test_batch)))
