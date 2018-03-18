import tensorflow as tf
from .base import Baseline
import numpy as np
import pdb

class ValueNetwork(Baseline):

    def __init__(self, env_spec, in_dim, hidden_dim, optimizer, session):

        # Placeholder Inputs
        self._observations = tf.placeholder(tf.float32, shape=[None, in_dim], name="observations")
        self._target = tf.placeholder(tf.float32, name="advantages")
        
        self._opt = optimizer
        self._sess = session

        h1 = tf.contrib.layers.fully_connected(self._observations, hidden_dim, tf.nn.tanhi)
        out = tf.contrib.layers.fully_connected(h1, 1, tf.identity)

        loss = tf.losses.mean_squared_error(self._target, out)
        train_op = self._opt.minimize(loss)
        predict_op = out

        # --------------------------------------------------
        # These operations (variables) are used when updating model
        self._predict_op = predict_op
        self._loss_op = loss
        self._train_op = train_op

    def predict(self, path):
        observations = path["observations"]
        pred = self._sess.run(self._predict_op, feed_dict={self._observations:observations})
        return np.reshape(pred, [-1])

    def fit(self, paths, verbose=False):
        observations = np.concatenate([path["observations"] for path in paths])
        targets = np.concatenate([path["returns"] for path in paths])
        loss, _ = self._sess.run([self._loss_op, self._train_op], feed_dict={self._observations:observations, 
                 self._target:targets})
        if verbose:
            print ("Value Loss:", loss)
