import os, re, time, glob
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer

from ops import *
from utils import *

import pickle
import tqdm

def dopickle(dict_, file):
    with open(file, 'wb') as fo:
        pickle.dump(dict_, fo)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_

# Use base classes to train the hallucinator
class HAL(object):
    def __init__(self,
                 sess,
                 model_name='HAL',
                 result_path='/home/cclin/few_shot_learning/hallucination_by_analogy/results',
                 fc_dim=512,
                 n_fine_class=80,
                 loss_lambda=10,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 l2scale=0.001):
        self.sess = sess
        self.model_name = model_name
        self.result_path = result_path
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        self.fc_dim = fc_dim
        self.n_fine_class = n_fine_class
        self.loss_lambda = loss_lambda
        self.bnDecay = bnDecay
        self.epsilon = epsilon
        self.l2scale = l2scale
    
    def build_model(self):
        self.triplet_features = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim*3], name='triplet_features')
        self.target_features = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim], name='target_features')
        self.fine_labels = tf.placeholder(tf.float32, shape=[None]+[self.n_fine_class], name='fine_labels')
        self.bn_train = tf.placeholder('bool', name='bn_train')
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        
        ### Used the classifier on the base classes learnt during representation learning
        ### to compute the classification loss of hallucinated features.
        self.bn_dense14 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_dense14')
        self.bn_dense15 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_dense15')
        print("build model started")
        self.hallucinated_features = self.build_hallucinator(self.triplet_features)
        self.logits = self.build_mlp(self.hallucinated_features)
        print("build model finished, define loss and optimizer")
        
        self.loss_mse = tf.reduce_mean((self.target_features - self.hallucinated_features)**2)
        self.loss_cls = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.fine_labels,
                                                                               logits=self.logits))
        self.loss = self.loss_lambda * self.loss_mse + self.loss_cls
        
        ### variables
        self.all_vars = tf.global_variables()
        self.all_vars_hal = [var for var in self.all_vars if 'hal' in var.name]
        self.all_vars_mlp = [var for var in self.all_vars if 'mlp' in var.name]
        self.trainable_vars = tf.trainable_variables()
        self.trainable_vars_hal = [var for var in self.trainable_vars if 'hal' in var.name]
        self.trainable_vars_mlp = [var for var in self.trainable_vars if 'mlp' in var.name]
        
        ### regularizers
        self.all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.used_regs = [reg for reg in self.all_regs if \
                          ('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name)]
        self.used_regs_hal = [reg for reg in self.all_regs if \
                              ('hal' in reg.name) and (('Matrix' in reg.name) or ('bias' in reg.name))]
        
        ### optimizers
        self.opt_hal = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                              beta1=0.5).minimize(self.loss+sum(self.used_regs_hal),
                                                                  var_list=self.trainable_vars_hal)
        #self.opt_all = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
        #                                      beta1=0.5).minimize(self.loss+sum(self.used_regs),
        #                                                          var_list=self.trainable_vars)
        
        ### Create model saver (keep the best 3 checkpoint)
        self.saver = tf.train.Saver(max_to_keep = 3)
        self.saver_hal = tf.train.Saver(var_list = self.all_vars_hal,
                                        max_to_keep = 3)
        self.saver_mlp = tf.train.Saver(var_list = self.all_vars_mlp,
                                        max_to_keep = 3)
        return [self.all_vars, self.trainable_vars, self.all_regs]
    
    ## Used the classifier on the base classes learnt during representation learning
    ## to compute the classification loss of hallucinated features.
    def build_mlp(self, input_):
        with tf.variable_scope('mlp', regularizer=l2_regularizer(self.l2scale)):
            ### Layer 14: dense with self.fc_dim neurons, BN, and ReLU
            self.dense14 = self.bn_dense14(linear(input_, self.fc_dim, name='dense14'), train=self.bn_train) ## [-1,self.fc_dim]
            self.relu14 = tf.nn.relu(self.dense14, name='relu14')
            ### Layer 15: dense with self.fc_dim neurons, BN, and ReLU
            self.dense15 = self.bn_dense15(linear(self.relu14, self.fc_dim, name='dense15'), train=self.bn_train) ## [-1,self.fc_dim]
            self.relu15 = tf.nn.relu(self.dense15, name='relu15')
            ### Layer 16: dense with self.n_fine_class neurons, softmax
            self.dense16 = linear(self.relu15, self.n_fine_class, add_bias=True, name='dense16') ## [-1,self.n_fine_class]
        return self.dense16
    
    ## "For our generator G, we use a three layer MLP with ReLU as the activation function" (Hariharan, 2017)
    def build_hallucinator(self, input_):
        with tf.variable_scope('hal', regularizer=l2_regularizer(self.l2scale)):
            ### Layer 1: dense with self.fc_dim neurons, ReLU
            self.dense1 = linear(input_, self.fc_dim, add_bias=True, name='dense1') ## [-1,self.fc_dim]
            self.relu1 = tf.nn.relu(self.dense1, name='relu1')
            ### Layer 2: dense with self.fc_dim neurons, ReLU
            self.dense2 = linear(self.relu1, self.fc_dim, add_bias=True, name='dense2') ## [-1,self.fc_dim]
            self.relu2 = tf.nn.relu(self.dense2, name='relu2')
            ### Layer 3: dense with self.fc_dim neurons, ReLU
            self.dense3 = linear(self.relu2, self.fc_dim, add_bias=True, name='dense3') ## [-1,self.fc_dim]
            self.relu3 = tf.nn.relu(self.dense3, name='relu3')
        return self.relu3
    
    def train(self,
              train_path,
              train_base_path,
              init_from=None, ## e.g., model_name (if None ==> train from scratch)
              cos_sim_threshold=0.0,
              bsize=32,
              learning_rate=5e-5,
              num_epoch=50,
              patience=10,
              ratio_for_dense=0.0):
        ### create a dedicated folder for this model
        if os.path.exists(os.path.join(self.result_path, self.model_name)):
            print('WARNING: the folder "{}" already exists!'.format(os.path.join(self.result_path, self.model_name)))
        else:
            os.makedirs(os.path.join(self.result_path, self.model_name))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'models'))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'models_hal'))
        
        ### Load training data (as a whole dictionary),
        ### and split each of them into training/validation by 80/20
        train_dict = unpickle(train_path)
        data_len = len(train_dict[b'fine_labels'])
        
        ### Use quadruplets with high cosine similarities (specified by 'cos_sim_threshold')
        qualified_indexes = [idx for idx in range(data_len) if train_dict[b'cos_sim'][idx] > cos_sim_threshold]
        triplet_feature = train_dict[b'triplet_features'][qualified_indexes]
        target_feature = train_dict[b'target_features'][qualified_indexes]
        fine_labels = [train_dict[b'fine_labels'][idx] for idx in qualified_indexes]
        data_len = len(fine_labels)
        print('After thresholdng, we have %d quadruplets' % data_len)

        triplet_feature_train = triplet_feature[0:int(data_len*0.8)]
        triplet_feature_valid = triplet_feature[int(data_len*0.8):int(data_len)]
        target_feature_train = target_feature[0:int(data_len*0.8)]
        target_feature_valid = target_feature[int(data_len*0.8):int(data_len)]
        nBatches = int(np.ceil(triplet_feature_train.shape[0] / bsize))
        nBatches_valid = int(np.ceil(triplet_feature_valid.shape[0] / bsize))
        ### one-hot, but need to consider the following error first:
        ### "IndexError: index 86 is out of bounds for axis 0 with size 80"
        ### Make a dictionary for {old_label: new_label} mapping, e.g., {0:0, 3:1, 5:2, 6:3, 7:4, ..., 99:79}
        ### such that all labels become 0~79
        label_mapping = {}
        ### [NOTE] Use fine_labels in train_base_dict to define label_mapping, since
        ###        some of base labels may not exist in the training data for hallucinator!
        train_base_dict = unpickle(train_base_path)
        for new_lb in range(self.n_fine_class):
            label_mapping[np.sort(list(set(train_base_dict[b'fine_labels'])))[new_lb]] = new_lb
        fine_labels_old = [int(s) for s in fine_labels[0:int(data_len*0.8)]]
        fine_labels_new = [label_mapping[old_lb] for old_lb in fine_labels_old]
        fine_labels_train = np.eye(self.n_fine_class)[fine_labels_new]
        fine_labels_old = [int(s) for s in fine_labels[int(data_len*0.8):int(data_len)]]
        fine_labels_new = [label_mapping[old_lb] for old_lb in fine_labels_old]
        fine_labels_valid = np.eye(self.n_fine_class)[fine_labels_new]
        
        ### Data indexes used to shuffle training order
        arr = np.arange(triplet_feature_train.shape[0])
        
        ### initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)
        
        ### load previously-trained linear classifier
        if init_from is not None:
            could_load, checkpoint_counter = self.load_mlp(init_from)
            if could_load:
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        else:
            print(" [@] train from scratch")
        
        ### Debug
        # print(label_mapping)
        # print('triplet_feature_train.shape = %s' % (triplet_feature_train.shape,))
        # print('triplet_feature_valid.shape = %s' % (triplet_feature_valid.shape,))
        # print('target_feature_train.shape = %s' % (target_feature_train.shape,))
        # print('target_feature_valid.shape = %s' % (target_feature_valid.shape,))
        # print('len(fine_labels_train) = %d' % len(fine_labels_train))
        # print('len(fine_labels_valid) = %d' % len(fine_labels_valid))
        # print('nBatches = %d' % nBatches)
        # print('nBatches_valid = %d' % nBatches_valid)

        ### main training loop
        loss_train = []
        loss_valid = []
        best_loss = 0
        stopping_step = 0
        for epoch in range(1, (num_epoch+1)):
            loss_train_batch = []
            loss_valid_batch = []
            #### shuffle training order for each epoch
            np.random.shuffle(arr)
            #print('training')
            for idx in tqdm.tqdm(range(nBatches)):
                batch_triplet_feature = triplet_feature_train[arr[idx*bsize:(idx+1)*bsize]]
                batch_target_feature = target_feature_train[arr[idx*bsize:(idx+1)*bsize]]
                batch_labels = fine_labels_train[arr[idx*bsize:(idx+1)*bsize]]
                #print(batch_labels.shape)
                _, loss = self.sess.run([self.opt_hal, self.loss],
                                        feed_dict={self.triplet_features: batch_triplet_feature,
                                                   self.target_features: batch_target_feature,
                                                   self.fine_labels: batch_labels,
                                                   self.bn_train: True,
                                                   self.keep_prob: 0.5,
                                                   self.learning_rate: learning_rate})
                loss_train_batch.append(loss)
            #### compute validation loss
            #print('validation')
            for idx in tqdm.tqdm(range(nBatches_valid)):
                batch_triplet_feature = triplet_feature_valid[idx*bsize:(idx+1)*bsize]
                batch_target_feature = target_feature_valid[idx*bsize:(idx+1)*bsize]
                batch_labels = fine_labels_valid[idx*bsize:(idx+1)*bsize]
                #print(batch_labels.shape)
                loss = self.sess.run(self.loss,
                                     feed_dict={self.triplet_features: batch_triplet_feature,
                                                self.target_features: batch_target_feature,
                                                self.fine_labels: batch_labels,
                                                self.bn_train: False,
                                                self.keep_prob: 1.0,})
                loss_valid_batch.append(loss)
            #### record training loss for each epoch (instead of each iteration)
            loss_train.append(np.mean(loss_train_batch))
            loss_valid.append(np.mean(loss_valid_batch))
            print('Epoch: %d, train loss: %f, valid loss: %f' % \
                  (epoch, np.mean(loss_train_batch), np.mean(loss_valid_batch)))
            
            #### save model if improvement, stop if reach patience
            current_loss = np.mean(loss_valid_batch)
            if epoch == 1:
                best_loss = current_loss
            else:
                if current_loss < best_loss: ## only monitor loss
                    best_loss = current_loss
                    self.saver.save(self.sess,
                                    os.path.join(self.result_path, self.model_name, 'models', self.model_name + '.model'),
                                    global_step=epoch)
                    self.saver_hal.save(self.sess,
                                        os.path.join(self.result_path, self.model_name, 'models_hal', self.model_name + '.model-hal'),
                                        global_step=epoch)
                    stopping_step = 0
                else:
                    stopping_step += 1
                print('stopping_step = %d' % stopping_step)
                if stopping_step >= patience:
                    print('stopping_step >= patience (%d), stop training' % patience)
                    break
        return [loss_train, loss_valid]
    
    def load_mlp(self, init_from, init_from_ckpt=None):
        ckpt = tf.train.get_checkpoint_state(init_from)
        if ckpt and ckpt.model_checkpoint_path:
            if init_from_ckpt is None:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            else:
                ckpt_name = init_from_ckpt
            self.saver_mlp.restore(self.sess, os.path.join(init_from, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
