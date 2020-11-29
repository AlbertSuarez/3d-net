import h5py

from tensorflow.python.keras.utils.np_utils import to_categorical

from src.config import TRAIN_INPUT_SIZE, DATASET_CATEGORIES, TRAIN_STRATEGY
from src.helper import log


def _cut(h5_read_object, cut_factor):
    return h5_read_object[:cut_factor] if cut_factor > 0 else h5_read_object[:]


def load(h5_file_path, h5_type, cut=0):
    log.info(f'Loading {h5_type}...')
    with h5py.File(h5_file_path, 'r') as f:
        x_train = _cut(f['x_train'], cut)
        if TRAIN_STRATEGY == 'relu':
            x_train = x_train.reshape(x_train.shape[0], TRAIN_INPUT_SIZE, TRAIN_INPUT_SIZE, 3)
        x_train = x_train.astype('float32') / 255
        y_train = to_categorical(_cut(f['y_train'], cut), num_classes=len(DATASET_CATEGORIES))
    log.info(f'{h5_type.capitalize()} items: {x_train.shape[0]}')
    return x_train, y_train
