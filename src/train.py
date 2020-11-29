import argparse
import os
import tensorflow as tf

from datetime import datetime

from src import loader
from src.config import DATASET_FOLDER_PREPROCESSED, DATASET_SUB_FOLDER_TRAINING, DATASET_CATEGORIES, \
    DATASET_FOLDER_WEIGHTS, TRAIN_EPOCHS, TRAIN_MODEL_FILE, TRAIN_INPUT_SIZE, TRAIN_STRATEGY


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

    x_train, y_train = loader.load(preprocessed_h5, DATASET_SUB_FOLDER_TRAINING)

    model = tf.keras.Sequential()
    if TRAIN_STRATEGY == 'relu':
        model.add(tf.keras.layers.Conv2D(256, (5, 5), activation='relu',
                                         input_shape=(TRAIN_INPUT_SIZE, TRAIN_INPUT_SIZE, 3)))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(512, (5, 5), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Flatten())
    else:
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
