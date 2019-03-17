import math
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
from data_server import load_dataset


def main():
    x_train, y_train, x_test, y_test, x_validate, y_validate, side_length = prepare_data('smile_warrior_dataset.csv')

    model = load_model("model.hdf5")
    model.load_weights('wagi_z_augmentacja/weights.03.hdf5')
    # model.load_weights('weights.03.hdf5')
    # model.compile('adadelta', 'mse', metrics=['accuracy'])

    score = model.evaluate(x_train, y_train, verbose=1)
    # Print test accuracy
    print('\n', 'Test accuracy:', score[1])
    print('\n', 'Test accuracy:', score[0])


def normalize(x_train, x_test, x_validate):
    """
    This module performs normalization of input images

    :param x_train:
        Input training data.

    :param x_test:
        Input testing data.

    :param x_validate:
        Input validation data.

    :return:
        Normalized input images.
    """
    return (x_train - 256 / 2) / 256, (x_test - 256 / 2) / 256, (x_validate - 256 / 2) / 256


def prepare_data(dataset):
    """
    This module prepares data for training and validation process.

    :param dataset:
        Dataset file path.

    :return:
        6 numpy arrays containing train, test and validate datasets for x and y.
        side_length containing height/width of processed images.
    """
    # Downloading and reshaping data
    x_train, y_train, x_test, y_test, x_validate, y_validate = load_dataset(dataset)
    side_length = int(math.sqrt(x_train.shape[1]))
    x_train = x_train.reshape(x_train.shape[0], side_length, side_length, 1)
    x_test = x_test.reshape(x_test.shape[0], side_length, side_length, 1)
    x_validate = x_validate.reshape(x_validate.shape[0], side_length, side_length, 1)

    x_train, x_test, x_validate = normalize(x_train, x_test, x_validate)

    # Preparing a proper format of output data
    y_train = np_utils.to_categorical(y_train, 2)
    y_test = np_utils.to_categorical(y_test, 2)
    y_validate = np_utils.to_categorical(y_validate, 2)

    return x_train, y_train, x_test, y_test, x_validate, y_validate, side_length


if __name__ == "__main__":
    main()
