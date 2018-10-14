import os, re, time, glob
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer

from sklearn.metrics import accuracy_score

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
        
        self.loss_mse = tf.reduce_mean((self.target_features - self.hallucinated_features)**2, axis=1)
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
        triplet_feature_train = train_dict[b'triplet_features'][0:int(data_len*0.8)]
        triplet_feature_valid = train_dict[b'triplet_features'][int(data_len*0.8):int(data_len)]
        target_feature_train = train_dict[b'target_features'][0:int(data_len*0.8)]
        target_feature_valid = train_dict[b'target_features'][int(data_len*0.8):int(data_len)]
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
        fine_labels = [int(s) for s in train_dict[b'fine_labels'][0:int(data_len*0.8)]]
        fine_labels_new = [label_mapping[old_lb] for old_lb in fine_labels]
        fine_labels_train = np.eye(self.n_fine_class)[fine_labels_new]
        fine_labels = [int(s) for s in train_dict[b'fine_labels'][int(data_len*0.8):int(data_len)]]
        fine_labels_new = [label_mapping[old_lb] for old_lb in fine_labels]
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

# FSL that takes extracted features as inputs directly.
# (Not allow fine-tuning the CNN-based feature extractor.)
# (No need to inherit from the "VGG" class.)
# (Allow hallucination!)
class FSL(object):
    def __init__(self,
                 sess,
                 model_name='FSL',
                 result_path='/home/cclin/few_shot_learning/hallucination_by_analogy/results',
                 fc_dim=512,
                 n_fine_class=100,
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
        self.bnDecay = bnDecay
        self.epsilon = epsilon
        self.l2scale = l2scale
    
    def build_model(self):
        self.features = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim], name='features')
        self.coarse_labels = tf.placeholder(tf.float32, shape=[None], name='coarse_labels')
        self.fine_labels = tf.placeholder(tf.float32, shape=[None]+[self.n_fine_class], name='fine_labels')
        self.bn_train = tf.placeholder('bool', name='bn_train')
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        
        ### Used the classifier on the base classes learnt during representation learning
        ### to compute the probability of the correct fine_label of hallucinated features.
        self.bn_dense14 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_dense14')
        self.bn_dense15 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_dense15')
        ### The classifier is implemented as a simple 3-layer MLP with batch normalization.
        ### Just like the one used in the VGG feature extractor. But it can be re-designed.
        self.bn_dense14_ = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_dense14_')
        self.bn_dense15_ = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_dense15_')
        print("build model started")
        self.logits = self.build_fsl_classifier(self.features)
        ### Also build the mlp.
        ### No need to define loss or optimizer since we only need foward-pass
        self.features_temp = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim], name='features_temp')
        self.logits_temp = self.build_mlp(self.features_temp)
        ### Also build the hallucinator.
        ### No need to define loss or optimizer since we only need foward-pass
        self.triplet_features = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim*3], name='triplet_features')
        self.hallucinated_features = self.build_hallucinator(self.triplet_features)
        print("build model finished, define loss and optimizer")
        
        ### Compute accuracy (optional)
        #self.outputs = tf.nn.softmax(self.dense16) ## [-1,self.n_fine_class]
        #self.pred = tf.argmax(self.outputs, axis=1) ## [-1,1]
        #self.acc = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.fine_labels), tf.float32))
        
        ### Define loss and training ops
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.fine_labels,
                                                                           logits=self.logits,
                                                                           name='loss'))
        
        ### variables
        self.all_vars = tf.global_variables()
        self.all_vars_hal = [var for var in self.all_vars if 'hal' in var.name]
        self.all_vars_fsl_cls = [var for var in self.all_vars if 'fsl_cls' in var.name]
        self.trainable_vars = tf.trainable_variables()
        self.trainable_vars_hal = [var for var in self.trainable_vars if 'hal' in var.name]
        self.trainable_vars_fsl_cls = [var for var in self.trainable_vars if 'fsl_cls' in var.name]
        
        ### regularizers
        self.all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.used_regs = [reg for reg in self.all_regs if \
                          ('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name)]
        self.used_regs_fsl_cls = [reg for reg in self.all_regs if \
                                  ('fsl_cls' in reg.name) and (('Matrix' in reg.name) or ('bias' in reg.name))]
        
        ### optimizers
        self.opt_fsl_cls = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                  beta1=0.5).minimize(self.loss+sum(self.used_regs_fsl_cls),
                                                                      var_list=self.trainable_vars_fsl_cls)
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
    ## to compute the probability of the correct fine_label of hallucinated features.
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
    
    ## The classifier is implemented as a simple 3-layer MLP with batch normalization.
    ## Just like the one used in the VGG feature extractor. But it can be re-designed.
    def build_fsl_classifier(self, input_):
        with tf.variable_scope('fsl_cls', regularizer=l2_regularizer(self.l2scale)):
            ### Layer 14: dense with self.fc_dim neurons, BN, and ReLU
            self.dense14_ = self.bn_dense14_(linear(input_, self.fc_dim, name='dense14_'), train=self.bn_train) ## [-1,self.fc_dim]
            self.relu14_ = tf.nn.relu(self.dense14_, name='relu14_')
            ### Layer 15: dense with self.fc_dim neurons, BN, and ReLU
            self.dense15_ = self.bn_dense15_(linear(self.relu14_, self.fc_dim, name='dense15_'), train=self.bn_train) ## [-1,self.fc_dim]
            self.relu15_ = tf.nn.relu(self.dense15_, name='relu15_')
            ### Layer 16: dense with self.n_fine_class neurons, softmax
            self.dense16_ = linear(self.relu15_, self.n_fine_class, add_bias=True, name='dense16_') ## [-1,self.n_fine_class]
        return self.dense16_

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
    
    def hallucinate(self,
                    seed_feature,
                    n_samples_needed,
                    train_base_path):
        #print(" [***] Hallucinator Load SUCCESS")
        ### Load training features and labels of the base classes
        ### (Take the first 80% since the rest are used for validation in the train() function)
        train_base_dict = unpickle(train_base_path)
        features_len = len(train_base_dict[b'fine_labels'])
        features_base_train = train_base_dict[b'features'][0:int(features_len*0.8)]
        fine_labels = [int(s) for s in train_base_dict[b'fine_labels'][0:int(features_len*0.8)]]
        labels_base_train = np.eye(self.n_fine_class)[fine_labels]
        
        ### Create a batch of size "n_samples_needed", with each row being consisted of
        ### (base_feature1, base_feature2, seed_feature), where base_feature1 and base_feature2
        ### are randomly selected from the same base class.
        input_features = np.empty([n_samples_needed, int(self.fc_dim*3)])
        all_possible_base_lbs = list(set(np.argmax(labels_base_train, axis=1)))
        for sample_count in range(n_samples_needed):
            #### (1) Randomly select a base class
            lb = np.random.choice(all_possible_base_lbs, 1)
            #### (2) Randomly select two samples from the above base class
            candidate_indexes = [idx for idx in range(labels_base_train.shape[0]) if np.argmax(labels_base_train[idx]) == lb]
            selected_indexes = np.random.choice(candidate_indexes, 2)
            #### (3) Concatenate (base_feature1, base_feature2, seed_feature) to form a row of the model input
            ####     Note that seed_feature has shape (1, fc_dim) already ==> no need np.expand_dims()
            input_features[sample_count,:] = np.concatenate((np.expand_dims(features_base_train[selected_indexes[0]], 0),
                                                             np.expand_dims(features_base_train[selected_indexes[1]], 0),
                                                             seed_feature), axis=1)
        
        ### Forward-pass
        features_hallucinated = self.sess.run(self.hallucinated_features,
                                              feed_dict={self.triplet_features: input_features})
        ### Choose the hallucinated features with high probability of the correct fine_label
        self.logits_temp = self.build_mlp(self.features_temp)
        logits_hallucinated = self.sess.run(self.logits_temp,
                                            feed_dict={self.features_temp: features_hallucinated})
        print('logits_hallucinated.shape: %s' % (logits_hallucinated.shape,))
        return features_hallucinated
    else:
        #print(" [***] Hallucinator Load FAIL, just repeat the seed_feature")
        return np.repeat(seed_feature, n_samples_needed, axis=0)
    
    def train(self,
              train_novel_path, ## train_novel_feat path (must be specified!)
              train_base_path, ## train_base_feat path (must be specified!)
              hal_from, ## e.g., hal_name (must given)
              mlp_from, ## e.g., mlp_name (must given)
              hal_from_ckpt=None, ## e.g., hal_name+'.model-1680' (can be None)
              mlp_from_ckpt=None, ## e.g., mlp_name+'.model-1680' (can be None)
              n_shot=1,
              n_min=20, ## minimum number of samples per training class ==> (n_min - n_shot) more samples need to be hallucinated
              n_top=5, ## top-n accuracy
              bsize=32,
              learning_rate=5e-5,
              num_epoch=50,
              patience=10):
        ### create a dedicated folder for this model
        if os.path.exists(os.path.join(self.result_path, self.model_name)):
            print('WARNING: the folder "{}" already exists!'.format(os.path.join(self.result_path, self.model_name)))
        else:
            os.makedirs(os.path.join(self.result_path, self.model_name))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'models'))
        
        ### Load training features (as two dictionaries) from both base and novel classes,
        ### and split each of them into training/validation by 80/20
        train_novel_dict = unpickle(train_novel_path)
        features_len = len(train_novel_dict[b'fine_labels'])
        features_novel_train = train_novel_dict[b'features'][0:int(features_len*0.8)]
        features_novel_valid = train_novel_dict[b'features'][int(features_len*0.8):int(features_len)]
        fine_labels = [int(s) for s in train_novel_dict[b'fine_labels'][0:int(features_len*0.8)]]
        labels_novel_train = np.eye(self.n_fine_class)[fine_labels]
        fine_labels = [int(s) for s in train_novel_dict[b'fine_labels'][int(features_len*0.8):int(features_len)]]
        labels_novel_valid = np.eye(self.n_fine_class)[fine_labels]
        train_base_dict = unpickle(train_base_path)
        features_len = len(train_base_dict[b'fine_labels'])
        features_base_train = train_base_dict[b'features'][0:int(features_len*0.8)]
        features_base_valid = train_base_dict[b'features'][int(features_len*0.8):int(features_len)]
        fine_labels = [int(s) for s in train_base_dict[b'fine_labels'][0:int(features_len*0.8)]]
        labels_base_train = np.eye(self.n_fine_class)[fine_labels]
        fine_labels = [int(s) for s in train_base_dict[b'fine_labels'][int(features_len*0.8):int(features_len)]]
        labels_base_valid = np.eye(self.n_fine_class)[fine_labels]
        
        ## load previous trained hallucinator
        could_load, checkpoint_counter = self.load_hal(hal_from, hal_from_ckpt)
        could_load, checkpoint_counter = self.load_mlp(mlp_from, mlp_from_ckpt)

        ### For the training split, use all base samples and randomly selected novel samples.
        if n_shot >= n_min:
            #### Hallucination not needed
            selected_indexes = []
            for lb in set(np.argmax(labels_novel_train, axis=1)):
                ##### Randomly select n-shot features from each class
                candidate_indexes_per_lb = [idx for idx in range(labels_novel_train.shape[0]) \
                                            if np.argmax(labels_novel_train[idx]) == lb]
                selected_indexes_per_lb = np.random.choice(candidate_indexes_per_lb, n_shot)
                selected_indexes.extend(selected_indexes_per_lb)
            features_novel_final = features_novel_train[selected_indexes]
            labels_novel_final = labels_novel_train[selected_indexes]
        else:
            #### Hallucination needed
            n_features_novel_final = int(n_min * len(set(np.argmax(labels_novel_train, axis=1))))
            features_novel_final = np.empty([n_features_novel_final, self.fc_dim])
            labels_novel_final = np.empty([n_features_novel_final, self.n_fine_class])
            lb_counter = 0
            for lb in set(np.argmax(labels_novel_train, axis=1)):
                ##### (1) Randomly select n-shot features from each class
                candidate_indexes_per_lb = [idx for idx in range(labels_novel_train.shape[0]) \
                                            if np.argmax(labels_novel_train[idx]) == lb]
                selected_indexes_per_lb = np.random.choice(candidate_indexes_per_lb, n_shot)
                selected_features_per_lb = features_novel_train[selected_indexes_per_lb]
                ##### (2) Randomly select a seed feature (from the above n-shot samples) for hallucination
                seed_index = np.random.choice(selected_indexes_per_lb, 1)
                seed_feature = features_novel_train[seed_index]
                ##### (3) Collect (n_shot) selected features and (n_min - n_shot) hallucinated features
                feature_hallucinated = self.hallucinate(seed_feature=seed_feature,
                                                        n_samples_needed=n_min-n_shot,
                                                        train_base_path=train_base_path)
                #print('feature_hallucinated.shape: %s' % (feature_hallucinated.shape,))
                features_novel_final[lb_counter*n_min:(lb_counter+1)*n_min,:] = \
                    np.concatenate((selected_features_per_lb, feature_hallucinated), axis=0)
                labels_novel_final[lb_counter*n_min:(lb_counter+1)*n_min,:] = np.eye(self.n_fine_class)[np.repeat(lb, n_min)]
                lb_counter += 1
        features_train = np.concatenate((features_novel_final, features_base_train), axis=0)
        print('features_train.shape: %s' % (features_train.shape,))
        fine_labels_train = np.concatenate((labels_novel_final, labels_base_train), axis=0)
        nBatches = int(np.ceil(features_train.shape[0] / bsize))
        
        ### For the validation split, just combine all base samples and all novel samples
        features_valid = np.concatenate((features_novel_valid, features_base_valid), axis=0)
        fine_labels_valid = np.concatenate((labels_novel_valid, labels_base_valid), axis=0)
        nBatches_valid = int(np.ceil(features_valid.shape[0] / bsize))
        
        ### Features indexes used to shuffle training order
        arr = np.arange(features_train.shape[0])
        
        ## initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)
        
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
                batch_features = features_train[arr[idx*bsize:(idx+1)*bsize]]
                batch_labels = fine_labels_train[arr[idx*bsize:(idx+1)*bsize]]
                #print(batch_labels.shape)
                _, loss, logits = self.sess.run([self.opt_mlp, self.loss, self.dense16],
                                                feed_dict={self.features: batch_features,
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
                batch_features = features_valid[idx*bsize:(idx+1)*bsize]
                batch_labels = fine_labels_valid[idx*bsize:(idx+1)*bsize]
                #print(batch_labels.shape)
                loss, logits = self.sess.run([self.loss, self.dense16],
                                             feed_dict={self.features: batch_features,
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
            print('           top-%d train accuracy: %f, top-%d valid accuracy: %f' % \
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
                    stopping_step = 0
                else:
                    stopping_step += 1
                print('stopping_step = %d' % stopping_step)
                if stopping_step >= patience:
                    print('stopping_step >= patience (%d), stop training' % patience)
                    break
        return [loss_train, loss_valid, acc_train, acc_valid]
    
    def inference(self,
                  test_novel_path, ## test_novel_feat path (must be specified!)
                  test_base_path=None, ## test_base_feat path (if None: close-world; else: open-world)
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
            
            #### load testing features
            if test_base_path:
                #### open-world
                test_novel_dict = unpickle(test_novel_path)
                test_base_dict = unpickle(test_base_path)
                features_test = np.concatenate((test_novel_dict[b'features'], test_base_dict[b'features']), axis=0)
                nBatches_test = int(np.ceil(features_test.shape[0] / bsize))
                #### one-hot
                fine_labels = [int(s) for s in test_novel_dict[b'fine_labels']+test_base_dict[b'fine_labels']]
                fine_labels_test = np.eye(self.n_fine_class)[fine_labels]
            else:
                #### close-world
                test_dict = unpickle(test_novel_path)
                features_test = test_dict[b'features']
                nBatches_test = int(np.ceil(features_test.shape[0] / bsize))
                #### one-hot
                fine_labels = [int(s) for s in test_dict[b'fine_labels']]
                fine_labels_test = np.eye(self.n_fine_class)[fine_labels]
            
            ### make prediction and compute accuracy
            loss_test_batch=[]
            acc_test_batch=[]
            top_n_acc_test_batch = []
            for idx in tqdm.tqdm(range(nBatches_test)):
                batch_features = features_test[idx*bsize:(idx+1)*bsize]
                batch_labels = fine_labels_test[idx*bsize:(idx+1)*bsize]
                loss, logits = self.sess.run([self.loss, self.dense16],
                                             feed_dict={self.features: batch_features,
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
    
    def load(self, init_from, init_from_ckpt=None):
        ckpt = tf.train.get_checkpoint_state(init_from)
        if ckpt and ckpt.model_checkpoint_path:
            if init_from_ckpt is None:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            else:
                ckpt_name = init_from_ckpt
            self.saver.restore(self.sess, os.path.join(init_from, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
    
    def load_hal(self, init_from, init_from_ckpt=None):
        ckpt = tf.train.get_checkpoint_state(init_from)
        if ckpt and ckpt.model_checkpoint_path:
            if init_from_ckpt is None:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            else:
                ckpt_name = init_from_ckpt
            self.saver_hal.restore(self.sess, os.path.join(init_from, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
    
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