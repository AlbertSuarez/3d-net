import h5py

from tensorflow.python.keras.utils.np_utils import to_categorical

from src.config import TRAIN_INPUT_SIZE, DATASET_CATEGORIES, TRAIN_STRATEGY
from src.helper import log


def load(h5_file_path, h5_type):
    log.info(f'Loading {h5_type}...')
    with h5py.File(h5_file_path, 'r') as f:
        if TRAIN_STRATEGY == 'relu':
            x_train = f['x_train'][:]
            x_train = x_train.reshape(x_train.shape[0], TRAIN_INPUT_SIZE, TRAIN_INPUT_SIZE, 3).astype('float32') / 255
        else:
            x_train = f['x_train'][:].astype('float32') / 255
        y_train = to_categorical(f['y_train'][:], num_classes=len(DATASET_CATEGORIES))
    log.info(f'{h5_type.capitalize()} items: {x_train.shape[0]}')
    return x_train, y_train
