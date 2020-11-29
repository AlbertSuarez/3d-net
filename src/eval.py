import argparse
import os
import h5py
import tensorflow as tf

from tensorflow.keras.utils import to_categorical

from src.config import DATASET_FOLDER_PREPROCESSED, DATASET_CATEGORIES, DATASET_FOLDER_WEIGHTS, \
    DATASET_SUB_FOLDER_VALIDATION, TRAIN_MODEL_FILE
from src.helper import log


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_name', type=str, required=True)
    parser.add_argument('--weights_folder', type=str, default=DATASET_FOLDER_WEIGHTS)
    parser.add_argument('--validation_h5', type=str,
                        default=os.path.join(DATASET_FOLDER_PREPROCESSED, f'{DATASET_SUB_FOLDER_VALIDATION}.h5'))
    return parser.parse_args()


def main(weights_name, weights_folder, validation_h5):
    model_path = os.path.join(weights_folder, weights_name, TRAIN_MODEL_FILE)
    assert os.path.isfile(model_path)

    with h5py.File(validation_h5, 'r') as f:
        x_train = f['x_train'][:].astype('float32') / 255
        y_train = to_categorical(f['y_train'][:], num_classes=len(DATASET_CATEGORIES))

    model = tf.keras.models.load_model(model_path)
    model.summary()
    loss, acc = model.evaluate(x_train, y_train, verbose=2)
    log.info('Restored model, accuracy: {:5.2f}%'.format(100 * acc))


if __name__ == '__main__':
    args = parse_args()
    main(args.weights_name, args.weights_folder, args.validation_h5)
