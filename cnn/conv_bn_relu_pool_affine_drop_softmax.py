import tensorflow as tf
import numpy as np
import collections as cols

from .wrappers import define_scope


class ConvBnReLUPoolAffineDropSoftmax(object):

    PLACEHOLDERS = ['keep_prob', 'is_training']

    def __init__(self, data, labels, input_HWC: tuple,
                 n_classes: int, n_conv_layers: int,
                 n_affine_layers: int, n_affine_neurons: int,
                 filter_params: dict, pool_params: dict, is_training: bool,
                 keep_prob: float, learning_rate=1e-3):

        self.data = tf.cast(data, tf.float32)
        self.labels = labels
        self.input_HWC = input_HWC

        self.n_classes = n_classes

        self.n_conv_layers = n_conv_layers
        self.n_affine_layers = n_affine_layers
        self.n_affine_neurons = n_affine_neurons

        self.filter_params = self._get_and_check_filter_params(n_conv_layers, filter_params)
        self.pool_params = self._get_and_check_pool_params(n_conv_layers, pool_params)
        self.keep_prob = keep_prob
        self.is_training = is_training

        self.learning_rate = learning_rate

        self.prediction
        self.optimize
        self.accuracy

    @define_scope
    def prediction(self):

        H, W, C = self.input_HWC
        conv_out = self._add_conv_layers(self.data, self.n_conv_layers, C,
                                         self.filter_params, self.pool_params)
        aff_out = self._add_affine_layers(conv_out, self.n_affine_layers,
                                          self.n_affine_neurons)

        scores = tf.identity(self._add_output_layer(aff_out, self.n_classes),
                             name='out')
        return scores

    @define_scope
    def optimize(self):

        total_loss = tf.losses.softmax_cross_entropy(tf.one_hot(self.labels, 10),
                                                     logits=self.prediction)
        mean_loss = tf.reduce_mean(total_loss)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        return optimizer.minimize(mean_loss, name='out')

    @define_scope
    def accuracy(self):

        correct = tf.equal(self.labels, tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(correct, tf.float32), name='out')

    def _get_and_check_filter_params(self, n_conv_layers: int, filter_params: dict):
        params = filter_params.copy()
        for k, v in params.items():
            if k != 'padding':
                if isinstance(v, cols.Iterable) and len(v) != n_conv_layers and len(v) > 1:
                    raise ValueError("The length of the field %d must be equal "
                                     "to the number of convolution layers n_conv_layers "
                                     "or be an integer" % k)
                if not isinstance(v, cols.Iterable):
                    params[k] = n_conv_layers * [v]
        return params

    def _get_and_check_pool_params(self, n_conv_layers: int, pool_params: dict):

        params = pool_params.copy()

        if 'padding' not in params.keys():
            params['padding'] = 'valid'

        for k, v in pool_params.items():
            if k != "padding":
                if isinstance(v, cols.Iterable) and len(v) != n_conv_layers and len(v) > 1:
                    raise ValueError("The length of the field must be equal "
                                     "to the number of convolution layers n_conv_layers "
                                     "or be an integer" % k)
                if not isinstance(v, cols.Iterable):
                    params[k] = n_conv_layers * [v]
        return params

    def _add_conv_layers(self, X, n_conv_layers: int, n_channels: int,
                         filter_params: dict, pool_params: dict):

        a_prev = X
        for l, n, sz, s, pool_sz, pool_s in zip(range(n_conv_layers),
                                                filter_params['n'],
                                                filter_params['size'],
                                                filter_params['strides'],
                                                pool_params['size'],
                                                pool_params['strides']):

            with tf.name_scope("ConvBlock_" + str(l)):

                h_conv = tf.layers.conv2d(a_prev, filters=n, kernel_size=sz, strides=(s, s),
                                          padding=filter_params['padding'],
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
                h_bn = tf.layers.batch_normalization(h_conv, training=self.is_training)
                h_relu = tf.nn.relu(h_bn)
                a_prev = tf.layers.max_pooling2d(h_relu, pool_size=pool_sz, strides=pool_s)

        n_out_neurons = int(np.prod(a_prev.shape[1:]))
        a_flatten = tf.reshape(a_prev, shape=[-1, n_out_neurons])

        h_drop = tf.nn.dropout(a_flatten, keep_prob=self.keep_prob)

        return h_drop

    def _add_affine_layers(self, conv_out_flatten, n_affine_layers: int, n_affine_neurons: int):

        a_prev = conv_out_flatten
        for l in range(n_affine_layers):

            with tf.name_scope("DenseBlock_" + str(l)):
                a_cur = tf.layers.dense(a_prev, n_affine_neurons, tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                a_prev = a_cur

        return tf.nn.dropout(a_prev, keep_prob=self.keep_prob)

    def _add_output_layer(self, affine_output, n_classes):

        with tf.name_scope("Output_Dense_Layer"):
            out = tf.layers.dense(affine_output, n_classes,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        return out
