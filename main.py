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


def train_net(cifar10_dir=r'./datasets/cifar-10-batches-py',
              # params_dir=r'./tunable_params',
              yaml_fname=None):

    X_train, y_train, X_val, y_val, X_test, y_test = \
        data_utils.get_CIFAR10_data(cifar10_dir)

    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)

    params_dir = os.path.dirname(yaml_fname)

    with open(yaml_fname, 'r') as params_yaml:

        print("params: " + yaml_fname)

        yaml_basename = os.path.basename(yaml_fname)
        model_dir = "./saved_models/model_" + yaml_basename[:-5]

        copy_params_yaml(yaml_basename, model_dir, params_dir)

        params = yaml.load(params_yaml)
        params['learning_rate'] = float(params['learning_rate'])
        if 'keep_prob' in params.keys():
            params['keep_prob'] = float(params['keep_prob'])
        params['input_HWC'] = X_train.shape[1:]
        params['n_classes'] = 10

        manager = cnn.Manager(params_as_placeholders=['keep_prob'])
        manager.create_model(cnn.ConvPoolDropReLUAffineDropAffineSoftmax,
                             X_train.shape[1:], **params)

        manager.train(X_train[:10], y_train[:10], X_val, y_val, X_test, y_test,
                      n_epoch=170, batch_size=64, verbose=True,
                      save_model_to_dir=model_dir,
                      save_model_every_epoch=10,
                      print_train_acc_every_batch=100)


if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument('-yaml', dest='yaml_fname')

    ap.add_argument('-cifar10_dir', dest='cifar10_dir',
                    action='store',
                    default=r'./datasets/cifar-10-batches-py')

    train_net(**vars(ap.parse_args()))
