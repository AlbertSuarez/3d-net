import argparse
import os
import h5py
import tensorflow as tf

from datetime import datetime
from tensorflow.keras.utils import to_categorical

from src.config import DATASET_FOLDER_PREPROCESSED, DATASET_SUB_FOLDER_TRAINING, DATASET_CATEGORIES, \
    DATASET_FOLDER_WEIGHTS, TRAIN_EPOCHS, TRAIN_MODEL_FILE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_folder', type=str, default=DATASET_FOLDER_WEIGHTS)
    parser.add_argument('--preprocessed_h5', type=str,
                        default=os.path.join(DATASET_FOLDER_PREPROCESSED, f'{DATASET_SUB_FOLDER_TRAINING}.h5'))
    parser.add_argument('--epochs', type=int, default=TRAIN_EPOCHS)
    return parser.parse_args()


def main(weights_folder, preprocessed_h5, epochs):
    weights_folder = os.path.join(weights_folder, datetime.utcnow().strftime('%Y-%m-%d__%H-%M-%S')) + os.sep
    os.makedirs(weights_folder)

    with h5py.File(preprocessed_h5, 'r') as f:
        x_train = f['x_train'][:].astype('float32') / 255
        y_train = to_categorical(f['y_train'][:], num_classes=len(DATASET_CATEGORIES))

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(len(DATASET_CATEGORIES), activation='sigmoid', input_shape=(x_train.shape[1],)))
    model.add(tf.keras.layers.Dense(len(DATASET_CATEGORIES), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=weights_folder, save_weights_only=True, verbose=1
    )
    model.fit(x_train, y_train, epochs=epochs, callbacks=[cp_callback])
    model.save(os.path.join(weights_folder, TRAIN_MODEL_FILE))


if __name__ == '__main__':
    args = parse_args()
    main(args.weights_folder, args.preprocessed_h5, args.epochs)
