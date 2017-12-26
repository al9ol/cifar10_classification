import os
import errno
import shutil
import yaml
import argparse

from datasets import data_utils
import cnn


def copy_params_yaml(fname, model_dir, params_dir):

    src_fname = os.path.join(params_dir, fname)
    dst_fname = os.path.join(model_dir, fname)

    if not os.path.exists(os.path.dirname(dst_fname)):
        try:
            os.makedirs(os.path.dirname(dst_fname))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise ValueError("Can't copy yaml-file")

    shutil.copyfile(src_fname, dst_fname)


def get_params_from_yaml(X_train, params_yaml,
                         params_to_float=('keep_prob', 'beta', 'learning_rate')):

    params = yaml.load(params_yaml)

    if params_to_float is not None:
        for p in params_to_float:
            if p in params.keys():
                params[p] = float(params[p])

    params['input_HWC'] = X_train.shape[1:]
    params['n_classes'] = 10

    return params


def train_net(cifar10_dir=r'./datasets/cifar-10-batches-py',
              yaml_fname=None):

    X_train, y_train, X_val, y_val, X_test, y_test = \
        data_utils.get_CIFAR10_data(cifar10_dir)

    with open(yaml_fname, 'r') as params_yaml:

        print("\n\n---\nparams: %s\n" % yaml_fname)

        yaml_basename = os.path.basename(yaml_fname)
        model_dir = "./saved_models/model_" + yaml_basename[:-5]
        params_dir = os.path.dirname(yaml_fname)

        copy_params_yaml(yaml_basename, model_dir, params_dir)

        params = get_params_from_yaml(X_train, params_yaml)

        manager = cnn.Manager(cnn.ConvPoolDropReLUAffineDropAffineL2Softmax)
        manager.create_model(X_train.shape[1:], **params)

        manager.train(X_train, y_train, X_val, y_val, X_test, y_test,
                      n_epoch=150, batch_size=64, verbose=True,
                      save_model_to_dir=model_dir,
                      save_model_every_epoch=0,
                      print_train_acc_every_batch=100)


if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument('-yaml', dest='yaml_fname')

    ap.add_argument('-cifar10_dir', dest='cifar10_dir',
                    action='store',
                    default=r'./datasets/cifar-10-batches-py')

    train_net(**vars(ap.parse_args()))
