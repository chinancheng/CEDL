import tensorflow as tf
import os
import random
import time
import numpy as np
import argparse
import math
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='convert data to feature')
    parser.add_argument('--source_path_feature', dest='source_path_feature', help='path of the feature map',
                        default='./feature_fc', type=str)
    parser.add_argument('--source_path_label', dest='source_path_label', help='path of the label',
                        default='./labels', type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='batch_size',
                        default='64', type=int)
    parser.add_argument('--mode', dest='mode', help='train/test',
                        default='train', type=str)
    parser.add_argument('--model_path', dest='model_path', help='path of the model',
                        default='./model', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

global args
args = parse_args()

class Data:
    def __init__(self, mode):
        self.feature_path = args.source_path_feature
        self.label_path = args.source_path_label
        self.batch_size = args.batch_size
        self.feature_list_train_ = []
        self.feature_list_validation_ = []
        self.feature_list_test = []
        self.mode = mode
        #split validation set from training set (training : validation = 19 : 1)
        self.feature_list = os.listdir(os.path.join(self.feature_path, self.mode))
        self.feature_list = [x for x in self.feature_list if x.find('Lhand') != -1]
        if mode == 'train':
            num = int(len(self.feature_list) * 0.95)
            self.train_data_len = num
            self.train_data_len = len(self.feature_list)
            self.validation_data_len = len(self.feature_list) - num
            self.feature_list_validation = self.feature_list[num:]
            self.feature_list_train = self.feature_list[:num]
            random.seed(time.time())
            random.shuffle(self.feature_list_train)
        elif mode == 'test':
            self.test_data_len = len(self.feature_list)

    def load_data(self, mode):
        if mode == 'train':
            if len(self.feature_list_train_) < self.batch_size:
                self.feature_list_train_ = self.feature_list_train
                random.seed(time.time())
                random.shuffle(self.feature_list_train_)
            if len(self.feature_list_train_) >= self.batch_size:
                batch_data_L = self.feature_list_train_[:self.batch_size]
                self.feature_list_train_ = self.feature_list_train_[self.batch_size:]
        elif mode == 'validation':
            if len(self.feature_list_validation_) < self.batch_size:
                self.feature_list_validation_ = self.feature_list_validation
            if len(self.feature_list_validation_) >= self.batch_size:
                batch_data_L = self.feature_list_validation_[:self.batch_size]
                self.feature_list_validation_ = self.feature_list_validation_[self.batch_size:]
        elif mode == 'test':
            if len(self.feature_list_test) < self.batch_size:
                self.feature_list_test = self.feature_list
            if len(self.feature_list_test) >= self.batch_size:
                batch_data_L = self.feature_list_test[:self.batch_size]
                self.feature_list_test = self.feature_list_test[self.batch_size:]
                    
        batch_data_head = []
        batch_label = []
        batch_data_R = []

        for path in batch_data_L:
            batch_data_head.append(path.replace('Lhand', 'head'))
            batch_data_R.append(path.replace('Lhand', 'Rhand'))
        for path_list in [batch_data_L, batch_data_R]: 
            for path in path_list:
                path = path.split('_')
                path_ = os.path.join(self.label_path, path[0])
                number = int(path[3].split('Image')[1].split('.npy')[0]) - 1
                if self.mode == 'test':
                    if path[0] == 'office' or path[0] == 'house':
                        path[1] = str(int(path[1]) + 3)             #test office/house 1.2.3 => 4.5.6
                    elif path[0] == 'lab':
                        path[1] = str(int(path[1]) + 4)             #test lab 1.2.3.4 => 5.6.7.8
                if path[2].find('Rhand') != -1:
                    fa = np.load(os.path.join(path_, 'FA_right' + path[1] + '.npy'))     #free/active
                    fa = int(fa[number]) 
                    ges = np.load(os.path.join(path_, 'ges_right' + path[1] + '.npy'))   #gesture
                    ges = int(ges[number])
                    obj = np.load(os.path.join(path_, 'obj_right' + path[1] + '.npy'))   #object
                    obj = int(obj[number])
                    batch_label.append([fa, ges, obj])
                elif path[2].find('Lhand') != -1:
                    fa = np.load(os.path.join(path_, 'FA_left' + path[1] + '.npy'))
                    fa = int(fa[number]) 
                    ges = np.load(os.path.join(path_, 'ges_left' + path[1] + '.npy')) 
                    ges = int(ges[number])
                    obj = np.load(os.path.join(path_, 'obj_left' + path[1] + '.npy'))
                    obj = int(obj[number])
                    batch_label.append([fa, ges, obj])
        batch_label = np.asarray(batch_label)
        batch_data_L = [np.load(os.path.join(self.feature_path, self.mode, x)) for x in batch_data_L]
        batch_data_R = [np.load(os.path.join(self.feature_path, self.mode, x)) for x in batch_data_R]
        batch_data_head = [np.load(os.path.join(self.feature_path, self.mode, x)) for x in batch_data_head]
        batch_data_L = np.asarray(batch_data_L)
        batch_data_R = np.asarray(batch_data_R)
        batch_data_head = np.asarray(batch_data_head)
        
        return batch_data_L, batch_label, batch_data_head, batch_data_R
        
    def get_data_length(self):
        if self.mode == 'train':
            return self.train_data_len, self.validation_data_len
        elif self.mode == 'test':
            return self.test_data_len

class Train:
    def __init__(self):
        self.batch_size = args.batch_size
        self.model_path = args.model_path
        self.epoch = 20
        self.start_learning_rate = 0.0001
        self.data_train = Data('train')
        self.data_test = Data('test')

    def train(self):
        model = Model()
        best_accuracy =0.0
        with tf.device('/gpu:0'):
            with tf.Graph().as_default():
                feature_Lhand, feature_Rhand, feature_head, label_fa, label_ges, label_obj, logits_fa, logits_ges, logits_obj, label_onehot_fa, label_onehot_ges, label_onehot_obj,_ = model.build(keep_prob=0.5)
                cross_entropy_fa = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_onehot_fa, logits=logits_fa, name='softmax_loss_fa'))
                tf.add_to_collection('losses_fa', cross_entropy_fa)
                cross_entropy_ges = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_onehot_ges, logits=logits_ges, name='softmax_loss_ges'))
                tf.add_to_collection('losses_ges', cross_entropy_ges)
                cross_entropy_obj = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_onehot_obj, logits=logits_obj, name='softmax_loss_obj'))
                tf.add_to_collection('losses_obj', cross_entropy_obj)
                loss_fa = tf.reduce_sum(tf.get_collection('losses_fa'))
                loss_ges = tf.reduce_sum(tf.get_collection('losses_ges'))
                loss_obj = tf.reduce_sum(tf.get_collection('losses_obj'))
                global_step = tf.Variable(0, trainable=False)
                learning_rate = self.start_learning_rate
                optimizer_1 = tf.train.AdamOptimizer(learning_rate=learning_rate)
                optimizer_2 = tf.train.AdamOptimizer(learning_rate=learning_rate)
                optimizer_3 = tf.train.AdamOptimizer(learning_rate=learning_rate)
                train_optimizer_fa = optimizer_1.minimize(loss_fa)
                train_optimizer_ges = optimizer_2.minimize(loss_ges)
                train_optimizer_obj = optimizer_3.minimize(loss_obj, global_step=global_step)
                tf.summary.scalar('loss_fa', loss_fa)
                tf.summary.scalar('loss_ges', loss_ges)
                tf.summary.scalar('loss_obj', loss_obj)
                summary = tf.summary.merge_all()
                with tf.Session() as sess:
                    if not os.path.exists(self.model_path):
                            os.makedirs(self.model_path)
                    summary_writer = tf.summary.FileWriter('./graph', sess.graph)
                    sess.run(tf.global_variables_initializer())
                    with tf.device('/cpu:0'):
                        saver = tf.train.Saver(max_to_keep=5)
                        if len(os.listdir(self.model_path)) > 0:
                            saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
                    print('Start training ......')
                    data_len_train, data_len_validation = self.data_train.get_data_length()
                    for i in range(int(data_len_train * self.epoch / self.batch_size)):
                        features_Lhand, labels, features_head, features_Rhand = self.data_train.load_data('train')
                        labels_fa = labels[:,0]
                        labels_ges = labels[:,1]
                        labels_obj = labels[:,2]
                        _, _, _, _loss_fa, _loss_ges, _loss_obj, step, summary_  = sess.run([train_optimizer_fa, train_optimizer_ges, train_optimizer_obj, loss_fa, loss_ges, loss_obj, global_step, summary], feed_dict={feature_Lhand:features_Lhand, feature_Rhand:features_Rhand, feature_head:features_head, label_fa:labels_fa, label_ges:labels_ges, label_obj:labels_obj})
                        summary_writer.add_summary(summary_, global_step=step)
                        if i % int(data_len_train/self.batch_size) == 0:
                            print 'epoch {0}' .format(math.floor(i*self.batch_size/data_len_train)+1)
                        if step % 10 == 0:
                            #print('Step: {0} Loss free/active: {1}' .format(step, _loss_fa))
                            #print('Step: {0} Loss gesture: {1}' .format(step, _loss_ges))
                            print('Step: {0} Loss object: {1}' .format(step, _loss_obj))
                        if step  % 100 == 0:
                            print('Saving models ......')
                            with tf.device('/cpu:0'):
                                saver.save(sess, os.path.join(self.model_path, 'model.ckpt'), global_step=step)
                            accuracy_fa_, accuracy_ges_, accuracy_obj_ = self.test(mode='validation', data_len=data_len_validation)
                            print('=> Validation accuracy free/active: {0}%' .format(accuracy_fa_ * 100))
                            print('=> Validation accuracy gesture: {0}%' .format(accuracy_ges_ * 100))
                            print('=> Validation accuracy object: {0}%' .format(accuracy_obj_ * 100))

    def test(self, mode, data_len=512):
        if args.mode == 'test':
           data_len = self.data_test.get_data_length() 
        model = Model()
        _accuracy = 0.0
        with tf.device('/gpu:0'):
            with tf.Graph().as_default():
                feature_Lhand, feature_Rhand, feature_head, label_fa, label_ges, label_obj, logits_fa, logits_ges, logits_obj, label_onehot_fa, label_onehot_ges, label_onehot_obj = model.build(keep_prob=1)
                prediction_fa = tf.argmax(logits_fa, axis=1)
                prediction_ges = tf.argmax(logits_ges, axis=1)
                prediction_obj = tf.argmax(logits_obj, axis=1)
                accuracy_fa, update_accuracy_fa = tf.metrics.accuracy(labels=label_fa, predictions=prediction_fa)
                tf.summary.scalar('accuracy_fa', accuracy_fa)
                accuracy_ges, update_accuracy_ges = tf.metrics.accuracy(labels=label_ges, predictions=prediction_ges)
                tf.summary.scalar('accuracy_ges', accuracy_ges)
                accuracy_obj, update_accuracy_obj = tf.metrics.accuracy(labels=label_obj, predictions=prediction_obj)
                tf.summary.scalar('accuracy_obj', accuracy_obj)
                tf.summary.scalar('accuracy',accuracy_obj)

                with tf.Session() as sess:
                    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
                    with tf.device('/cpu:0'):
                        restorer = tf.train.Saver()
                        restorer.restore(sess, tf.train.latest_checkpoint(self.model_path))
                    for _ in range(int(data_len / self.batch_size)):
                        if mode == 'test': 
                            features_Lhand, labels, features_head, features_Rhand = self.data_test.load_data('test')
                        elif mode == 'train':
                            features_Lhand, labels, features_head, features_Rhand = self.data_train.load_data('train')
                        elif mode == 'validation':
                            features_Lhand, labels, features_head, features_Rhand = self.data_train.load_data('validation')
                        labels_fa = labels[:,0]
                        labels_ges = labels[:,1]
                        labels_obj = labels[:,2]
                        _, _, _  = sess.run([update_accuracy_fa, update_accuracy_ges, update_accuracy_obj], feed_dict={feature_Lhand:features_Lhand, feature_Rhand:features_Rhand, feature_head:features_head, label_fa:labels_fa, label_ges:labels_ges, label_obj:labels_obj})
                        _accuracy_fa, _accuracy_ges, _accuracy_obj = sess.run([accuracy_fa, accuracy_ges, accuracy_obj])
                    return _accuracy_fa,  _accuracy_ges,  _accuracy_obj
        
class Model:
    def __init__(self):
        self.batch_size = args.batch_size
        self.data_dict = np.load('./vgg19.npy', encoding='latin1').item()

    def build(self, keep_prob):
        feature_Lhand = tf.placeholder(tf.float32, shape=[self.batch_size, 4096], name='feature_Lhand')
        feature_Rhand = tf.placeholder(tf.float32, shape=[self.batch_size, 4096], name='feature_Rhand')
        feature_head = tf.placeholder(tf.float32, shape=[self.batch_size, 4096], name='feature_head_input')
        label_fa = tf.placeholder(tf.int32, shape=[2*self.batch_size], name='label_fa')
        label_onehot_fa = tf.one_hot(label_fa, depth=2)
        label_ges = tf.placeholder(tf.int32, shape=[2*self.batch_size], name='label_ges')
        label_onehot_ges = tf.one_hot(label_ges, depth=13)
        label_obj = tf.placeholder(tf.int32, shape=[2*self.batch_size], name='label_obj')
        label_onehot_obj = tf.one_hot(label_obj, depth=24)
        fc1 = tf.concat([feature_Lhand, feature_Rhand, feature_head], axis=1)
        with tf.variable_scope('fc2'):
            fc2 = self.fc_layer(fc1, [4096*3, 4096], [4096], wd=0.0, _class='all')
            fc2 = tf.nn.relu(fc2)
        with tf.variable_scope('dropout2'):
            fc2 = tf.nn.dropout(fc2, keep_prob=keep_prob)
        with tf.variable_scope('fc3_fa_L'):
            fc3_fa_L = self.fc_layer(fc2, [4096, 2], [2], wd=0.0001, _class='fa')
        with tf.variable_scope('fc3_ges_L'):
            fc3_ges_L = self.fc_layer(fc2, [4096, 13], [13], wd=0.0001, _class='ges')
        with tf.variable_scope('fc3_obj_L'):
            fc3_obj_L = self.fc_layer(fc2, [4096, 24], [24], wd=0.0001, _class='obj')
        with tf.variable_scope('fc3_fa_R'):
            fc3_fa_R = self.fc_layer(fc2, [4096, 2], [2], wd=0.0001, _class='fa')
        with tf.variable_scope('fc3_ges_R'):
            fc3_ges_R = self.fc_layer(fc2, [4096, 13], [13], wd=0.0001, _class='ges')
        with tf.variable_scope('fc3_obj_R'):
            fc3_obj_R = self.fc_layer(fc2, [4096, 24], [24], wd=0.0001, _class='obj')
        fc3_fa = tf.concat([fc3_fa_L, fc3_fa_R], 0)
        fc3_ges = tf.concat([fc3_ges_L, fc3_ges_R], 0)
        fc3_obj = tf.concat([fc3_obj_L, fc3_obj_R], 0)

        return feature_Lhand, feature_Rhand, feature_head, label_fa, label_ges, label_obj, fc3_fa, fc3_ges, fc3_obj, label_onehot_fa, label_onehot_ges, label_onehot_obj
       


    def fc_layer(self, _input, kernel_shape, bias_shape, wd, _class):
        w = tf.get_variable('weights', shape=kernel_shape, initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('bias', shape=bias_shape, initializer=tf.constant_initializer(0.0))
        weight_decay = tf.multiply(tf.nn.l2_loss(w), wd, name='weight_loss')
        
        if _class == 'all':       
            tf.add_to_collection('losses_fa', weight_decay)
            tf.add_to_collection('losses_ges', weight_decay)
            tf.add_to_collection('losses_obj', weight_decay)
        elif _class == 'fa':
            tf.add_to_collection('losses_fa', weight_decay)
        elif _class == 'ges':
            tf.add_to_collection('losses_ges', weight_decay)
        elif _class == 'obj':
            tf.add_to_collection('losses_obj', weight_decay)
        
        return tf.nn.xw_plus_b(x=_input, weights=w, biases=b)

 
        


if __name__ == '__main__':
    print args
    train = Train()
    if args.mode == 'train':
        train.train()
    elif args.mode == 'test':
        accuracy_fa, accuracy_ges, accuracy_obj = train.test(mode='test')
        #print('=>Testing accuracy free/active: {0}%' .format(accuracy_fa * 100))
        #print('=>Testing accuracy gesture: {0}%' .format(accuracy_ges * 100))
        print('=>Testing accuracy object: {0}%' .format(accuracy_obj * 100))
  
