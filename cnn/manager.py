import os
import errno
import pathlib
import copy
import yaml

import tensorflow as tf
import numpy as np

from . import abstract_model as amdl


class Manager(object):

    def __init__(self, params_as_placeholders):

        self.placeholder_names = []

        if params_as_placeholders is not None:
            self.params_as_placeholders = params_as_placeholders
        else:
            self.params_as_placeholders = []

    def create_model(self, ModelClass, HWC, **model_params):

        self.data = tf.placeholder(tf.float32, [None, *HWC], name='data')
        self.labels = tf.placeholder(tf.int64, [None], name='labels')

        self.model_params = copy.deepcopy(model_params)
        self._define_placeholders()

        self.model = ModelClass(self.data, self.labels, **self.model_params)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _define_placeholders(self):

        for ph in self.params_as_placeholders:
            if ph in self.model_params.keys():

                if ph not in self.placeholder_names:
                    self.placeholder_names.append(ph)

                setattr(self, ph + '_val', self.model_params[ph])
                setattr(self, ph, tf.placeholder(tf.float32, name=ph))
                self.model_params[ph] = getattr(self, ph)

    def _get_feed_dict(self, X, y, mode='train'):

        feed_dict = {self.data: X, self.labels: y}

        for ph in self.placeholder_names:
            if mode == 'test' and ph == 'keep_prob':
                feed_dict[self.keep_prob] = 1.0
            else:
                feed_dict[getattr(self, ph)] = getattr(self, ph + '_val')

        return feed_dict

    def train(self, X, y, X_val, y_val, X_test, y_test,
              n_epoch, batch_size, verbose=True,
              yaml_placeholders_fname="placeholders.yaml",
              save_model_to_dir="saved_model", save_model_every_epoch=1,
              best_model_dir="best", print_train_error_every_batch=100):

        n_batches = X.shape[0] // batch_size
        indices = np.arange(X.shape[0])

        best_error = 0

        for i_epoch in range(n_epoch):

            if verbose: print(f"epoch: {i_epoch}")

            for i_batch in range(n_batches):

                i_begin = i_batch * batch_size
                batch_slice = slice(i_begin, i_begin + batch_size)

                X_batch = X[indices[batch_slice]]
                y_batch = y[indices[batch_slice]]

                self.sess.run(self.model.optimize, self._get_feed_dict(X_batch, y_batch, 'train'))

                if verbose and i_batch % print_train_error_every_batch == 0:

                    batch_error = self.sess.run(self.model.error, self._get_feed_dict(X_batch, y_batch, 'test'))
                    print(f"\tbatch: {i_batch}\terror: {batch_error}")

            if save_model_every_epoch > 0 and i_epoch % save_model_every_epoch == 0:

                folder = os.path.join(save_model_to_dir, str(i_epoch))
                self.save_model(folder, "model", yaml_placeholders_fname)

            error = self.sess.run(self.model.error, self._get_feed_dict(X_val, y_val, 'test'))

            if i_epoch == 0 or error < best_error:

                best_error = error
                folder = os.path.join(save_model_to_dir, best_model_dir)
                self.save_model(folder, "model", yaml_placeholders_fname)

            if verbose:
                print(f"epoch: {i_epoch}\tbest error: {best_error:1.4f}\tval error: {error:1.4f}")

        error = self.sess.run(self.model.error, self._get_feed_dict(X_test, y_test, 'test'))
        if verbose: print(f"\n\ntest error: {error}")

    def save_model(self, dir, model_fname, yaml_placeholders_fname):

        saver = tf.train.Saver()
        saver.save(self.sess, os.path.join(dir, model_fname))
        self._save_placeholders(dir, yaml_placeholders_fname)

    def load_model(self, dir, model_name, meta_file=r".meta"):

        self.sess = tf.Session()

        saver = tf.train.import_meta_graph(os.path.join(dir, model_name + meta_file))
        saver.restore(self.sess, tf.train.latest_checkpoint(dir))

        graph = tf.get_default_graph()

        self.data = graph.get_tensor_by_name('data:0')
        self.labels = graph.get_tensor_by_name('labels:0')

        self.model = self._get_model_from_graph(graph)
        self._load_placeholders(dir, 'placeholders.yaml', graph)

    def _get_model_from_graph(self, graph):

        model = amdl.AbstractModel()

        setattr(model, 'prediction', graph.get_tensor_by_name('prediction/prediction_out:0'))
        setattr(model, 'optimize', graph.get_operation_by_name('optimize/optimize_out'))
        setattr(model, 'error', graph.get_tensor_by_name('error/error_out:0'))

        return model

    def _save_placeholders(self, dir, yaml_fname):

        params_to_save = {}
        for th in self.placeholder_names:
            params_to_save[th] = getattr(self, th + "_val")

        full_fname = os.path.join(dir, yaml_fname)
        if not os.path.exists(os.path.dirname(full_fname)):
            try:
                os.makedirs(os.path.dirname(full_fname))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise ValueError("Can not create file: " + full_fname)

        with open(full_fname, 'w') as fout:
            yaml.dump(params_to_save, fout)

    def _load_placeholders(self, dir, yaml_fname, graph):

        yaml_cache = pathlib.Path(os.path.join(dir, yaml_fname))
        if yaml_cache.is_file():
            with open(os.path.join(dir, yaml_fname), 'r') as fin:

                placeholders = yaml.load(fin)
                for k, v in placeholders.items():

                    setattr(self, k, graph.get_tensor_by_name(k + ":0"))
                    setattr(self, k + "_val", v)

                    if k not in self.placeholder_names:
                        self.placeholder_names.append(k)

    def __del__(self):

        if hasattr(self, 'sess'):
            self.sess.close()
