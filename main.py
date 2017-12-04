from datasets import data_utils
from datasets.sample_preprocessing import get_stratified_subsample
import cnn


def train_new_net(cifar10_dir='./datasets/cifar-10-batches-py'):

    X_train, y_train, X_val, y_val, X_test, y_test = \
        data_utils.get_CIFAR10_data(cifar10_dir)

    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)

    params = {'input_HWC': X_train.shape[1:],
              'n_classes': 10,
              'n_conv_layers': 3,
              'n_affine_layers': 2, 'n_affine_neurons': 128,
              'filter_params': {"n": [32, 64, 128], "strides": 1, "size": 3,
                                "padding": 'VALID'},
              'pool_params': {"strides": 1, "size": 2},
              'keep_prob': 0.7,
              'learning_rate': 1e-4}


    manager = cnn.Manager(params_as_placeholders=['keep_prob'])
    manager.create_model(cnn.ConvReLUPoolDropAffineSoftmax, X_train.shape[1:], **params)

    manager.train(X_train, y_train, X_val, y_val, X_test, y_test,
                  n_epoch=100, batch_size=64, verbose=True,
                  save_model_to_dir="saved_model", save_model_every_epoch=10,
                  print_train_acc_every_batch=100)

    # manager_restored = cnn.Manager(params_as_placeholders=['keep_prob'])
    # manager_restored.load_model(dir="saved_model/best", model_name='model')
    # manager_restored.train(X_train[::50], y_train[::50], X_val, y_val, X_test, y_test,
    #                        n_epoch=100, batch_size=64, verbose=True,
    #                        save_model_to_dir="saved_model", save_model_every_epoch=1,
    #                        print_train_error_every_batch=5)


if __name__ == "__main__":
    train_new_net()
