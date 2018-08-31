import os, re, time, glob
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import skimage
import skimage.transform
import skimage.io

import pandas as pd
from sklearn.metrics import accuracy_score

from ops import *
from utils import *

class MATCH_NET(object):
    def __init__(self,
                 sess,
                 model_name='MATCH_NET',
                 result_path='test1',
                 x_dim=28,
                 y_dim=5,
                 n_samples_per_class=1,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 tie=True):
        self.sess = sess
        self.model_name = model_name
        self.result_path = result_path
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_samples_per_class = n_samples_per_class
        self.n_samples = self.y_dim * self.n_samples_per_class
        self.bnDecay = bnDecay
        self.epsilon = epsilon
        self.tie = tie
    
    def build_model(self):
        ## placeholder for hyper-parameters
        self.bn_train = tf.placeholder('bool')
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        ## placeholder for data
        self.x_i = tf.placeholder(tf.float32, shape=[None, self.n_samples, self.x_dim, self.x_dim, 1])
        self.y_i_ind = tf.placeholder(tf.int32, shape=[None, self.n_samples])
        self.y_i = tf.one_hot(self.y_i_ind, self.y_dim)
        self.x_hat = tf.placeholder(tf.float32, shape=[None, self.x_dim, self.x_dim, 1])
        self.y_hat_ind = tf.placeholder(tf.int32, shape=[None])
        self.y_hat = tf.one_hot(self.y_hat_ind, self.y_dim)
        
        ## batch normalization
        self.bn0 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn0')
        self.bn1 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn1')
        self.bn2 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn2')
        self.bn3 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn3')
        
        cos_sim_list = []
        varscope = 'encode_x'
        self.x_hat_encode = self.conv_net(self.x_hat, scope=varscope, bn_train=self.bn_train) # [-1, 64]
        if not self.tie:
            varscope = 'encode_x_i'
        for i in range(self.n_samples):
            x_i_encode = self.conv_net(self.x_i[:,i,:,:,:], scope=varscope, bn_train=self.bn_train, reuse=(self.tie or i > 0)) # [-1, 64]
            dotted = tf.expand_dims(tf.reduce_sum(tf.multiply(self.x_hat_encode, x_i_encode), axis=1), axis=1) # [-1, 1]
            x_i_inv_mag = tf.rsqrt(tf.clip_by_value(tf.reduce_sum(tf.square(x_i_encode), 1, keep_dims=True), self.epsilon, float("inf")))
            cos_sim_list.append(dotted * x_i_inv_mag)
        cos_sim = tf.concat(cos_sim_list, axis=1) # [-1, self.n_samples]
        
        weighting = tf.nn.softmax(cos_sim)
        label_prob = tf.squeeze(tf.matmul(tf.expand_dims(weighting, 1), self.y_i))
        top_k = tf.nn.in_top_k(label_prob, self.y_hat_ind, 1)
        self.acc = tf.reduce_mean(tf.to_float(top_k))
        correct_prob = tf.reduce_sum(tf.log(tf.clip_by_value(label_prob, self.epsilon, 1.0)) * self.y_hat, 1)
        self.loss = tf.reduce_mean(-correct_prob, 0)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(self.loss)
    
    def conv_net(self, _input, scope, bn_train, reuse=False, stop_grad=False):
        with tf.variable_scope(scope) as varscope:
            if reuse:
                varscope.reuse_variables()
            conv0 = tf.nn.relu(self.bn0(conv2d(_input, output_dim=64, add_bias=False,
                                               k_h=3, k_w=3, d_h=1, d_w=1, name='conv0'), train=bn_train))
            pool0 = self.max_pool(conv0, name='pool0', pad='VALID') # [-1, 14, 14, 64]
            conv1 = tf.nn.relu(self.bn1(conv2d(pool0, output_dim=64, add_bias=False,
                                               k_h=3, k_w=3, d_h=1, d_w=1, name='conv1'), train=bn_train))
            pool1 = self.max_pool(conv1, name='pool1', pad='VALID') # [-1, 7, 7, 64]
            conv2 = tf.nn.relu(self.bn2(conv2d(pool1, output_dim=64, add_bias=False,
                                               k_h=3, k_w=3, d_h=1, d_w=1, name='conv2'), train=bn_train))
            pool2 = self.max_pool(conv2, name='pool2', pad='VALID') # [-1, 3, 3, 64]
            conv3 = tf.nn.relu(self.bn3(conv2d(pool2, output_dim=64, add_bias=False,
                                               k_h=3, k_w=3, d_h=1, d_w=1, name='conv3'), train=bn_train))
            pool3 = self.max_pool(conv3, name='pool3', pad='VALID') # [-1, 1, 1, 64]
            output = tf.squeeze(pool3, [1,2]) # [-1, 64]
            if stop_grad:
                return tf.stop_gradient(output)
            else:
                return output
    
    def max_pool(self, bottom, name, pad='SAME'):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=pad, name=name)
    
    def train(self,
              list_train,
              list_test,
              drawing_per_char=20,
              nEpochs=1e6,
              bsize=32,
              learning_rate_start=1e-3,
              patience=10):
        ## create a dedicated folder for this model
        if os.path.exists(os.path.join(self.result_path, self.model_name)):
            print('WARNING: the folder "{}" already exists!'.format(os.path.join(self.result_path, self.model_name)))
        else:
            os.makedirs(os.path.join(self.result_path, self.model_name))
        
        ## initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)
        
        for epoch in range(1, (int(nEpochs)+1)):
            mb_x_i, mb_y_i, mb_x_hat, mb_y_hat = self.get_minibatch(list_used=list_train,
                                                                    drawing_per_char=drawing_per_char,
                                                                    mb_dim=bsize)
            _, train_loss, train_acc = self.sess.run([self.train_op, self.loss, self.acc],
                                               feed_dict={self.bn_train: True,
                                                          self.learning_rate: learning_rate_start,
                                                          self.x_i: mb_x_i,
                                                          self.y_i_ind: mb_y_i,
                                                          self.x_hat: mb_x_hat,
                                                          self.y_hat_ind: mb_y_hat})
            if epoch % int(1e2) == 0:
                mb_x_i, mb_y_i, mb_x_hat, mb_y_hat = self.get_minibatch(list_used=list_test,
                                                                        drawing_per_char=drawing_per_char,
                                                                        mb_dim=bsize)
                test_loss, test_acc, x_hat_encode = self.sess.run([self.loss, self.acc, self.x_hat_encode],
                                                    feed_dict={self.bn_train: False,
                                                               self.x_i: mb_x_i,
                                                               self.y_i_ind: mb_y_i,
                                                               self.x_hat: mb_x_hat,
                                                               self.y_hat_ind: mb_y_hat})
                #print('x_hat_encode.shape = %s' % (x_hat_encode.shape,))
                print('epoch:', epoch,
                      '| train_loss:', train_loss,
                      '| train_acc:', train_acc,
                      '| test_loss:', test_loss,
                      '| test_acc:', test_acc)
    
    def inference(self):
        pass
    
    def get_minibatch(self,
                      list_used,
                      drawing_per_char=20,
                      mb_dim=32):
        mb_x_i = np.zeros((mb_dim, self.n_samples, self.x_dim, self.x_dim, 1))
        mb_y_i = np.zeros((mb_dim, self.n_samples), dtype=np.int)
        mb_x_hat = np.zeros((mb_dim, self.x_dim, self.x_dim, 1))
        mb_y_hat = np.zeros((mb_dim,), dtype=np.int)
        for i in range(mb_dim):
            ind = 0
            pinds = np.random.permutation(self.n_samples)
            class_indexes = np.random.choice(list_used.shape[0], self.y_dim, False)
            x_hat_class = np.random.randint(self.y_dim)
            for j, cur_class in enumerate(class_indexes): #each class
                example_inds = np.random.choice(drawing_per_char, self.n_samples_per_class, False)
                drawing_list = np.sort(glob.glob(os.path.join(list_used[cur_class][0], '*.png')))
                for eind in example_inds:
                    cur_data = transform(get_image(drawing_list[eind]),
                                         105, 105, resize_height=self.x_dim, resize_width=self.x_dim, crop=False)
                    mb_x_i[i, pinds[ind], :, :, 0] = np.rot90(cur_data, np.random.randint(4))
                    mb_y_i[i, pinds[ind]] = j
                    ind +=1
                if j == x_hat_class:
                    eval_idx = np.random.choice(drawing_per_char, 1, False)[0]
                    cur_data = transform(get_image(drawing_list[eval_idx]),
                                         105, 105, resize_height=self.x_dim, resize_width=self.x_dim, crop=False)
                    mb_x_hat[i, :, :, 0] = np.rot90(cur_data, np.random.randint(4))
                    mb_y_hat[i] = j
        return mb_x_i, mb_y_i, mb_x_hat, mb_y_hat
    
    