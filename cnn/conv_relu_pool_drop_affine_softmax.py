import tensorflow as tf
import numpy as np
import collections as cols

from .wrappers import define_scope

# TODO: add L2-regularization
# TODO: add possibility to change an optimization method


class ConvReLUPoolDropAffineSoftmax(object):

    def __init__(self, data, labels, input_HWC: tuple,
                 n_classes: int, n_conv_layers: int,
                 n_affine_layers: int, n_affine_neurons: int,
                 filter_params: dict, pool_params: dict,
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

        self.learning_rate = learning_rate

        self.prediction
        self.optimize
        self.accuracy

    @define_scope
    def prediction(self):

        H, W, C = self.input_HWC
        conv_out = self._add_conv_layers(self.data, self.n_conv_layers, C,
                                         self.filter_params, self.pool_params)
        aff_out = self._add_affine_layers(conv_out, self.n_affine_layers, self.n_affine_neurons)

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

        mistakes = tf.equal(self.labels, tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32), name='out')

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
        for k, v in pool_params.items():
            if isinstance(v, cols.Iterable) and len(v) != n_conv_layers and len(v) > 1:
                raise ValueError("The length of the field %d must be equal "
                                 "to the number of convolution layers n_conv_layers "
                                 "or be an integer" % k)
            if not isinstance(v, cols.Iterable):
                params[k] = n_conv_layers * [v]
        return params

    def _add_conv_layers(self, X, n_conv_layers: int, n_channels: int,
                         filter_params: dict, pool_params: dict):
        """[conv-relu-pool-drop]xN"""

        a_prev = X
        depth = n_channels

        for l, n, sz, s, pool_sz, pool_s in zip(range(n_conv_layers),
                                                filter_params['n'],
                                                filter_params['size'],
                                                filter_params['strides'],
                                                pool_params['size'],
                                                pool_params['strides']):

            W_cur = tf.get_variable("Wconv" + str(l + 1), shape=[sz, sz, depth, n],
                                    initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
            b_cur = tf.get_variable("bconv" + str(l + 1), shape=[n], initializer=tf.zeros_initializer)

            a_cur = tf.nn.conv2d(a_prev, W_cur, strides=[1, s, s, 1], padding=filter_params['padding']) + b_cur
            h_cur = tf.nn.relu(a_cur)

            h_pool = tf.layers.max_pooling2d(h_cur, pool_size=pool_sz, strides=pool_s)
            h_drop = tf.nn.dropout(h_pool, keep_prob=self.keep_prob)

            a_prev = h_drop
            depth = n

        return a_prev

    def _add_affine_layers(self, conv_out, n_affine_layers: int, n_affine_neurons: int):

        n_neurons = int(np.prod(conv_out.shape[1:]))
        a_prev = tf.reshape(conv_out, shape=[-1, n_neurons])

        for l in range(n_affine_layers):

            w_cur = tf.get_variable("Waff" + str(l + 1), shape=[n_neurons, n_affine_neurons],
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            b_cur = tf.get_variable("baff" + str(l + 1), shape=[n_affine_neurons],
                                    initializer=tf.zeros_initializer)

            h = tf.matmul(a_prev, w_cur) + b_cur
            a_cur = tf.nn.relu(h, "relu_aff" + str(l + 1))

            n_neurons = n_affine_neurons
            a_prev = a_cur

        return a_prev

    def _add_output_layer(self, affine_output, n_classes):

        w = tf.get_variable("Wout", shape=[affine_output.shape[1], n_classes],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        b = tf.get_variable("bout", shape=[n_classes], initializer=tf.zeros_initializer)

        return affine_output @ w + b
