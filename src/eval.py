import argparse
import os
import numpy as np
import tensorflow as tf

from src import loader
from src.config import DATASET_FOLDER_PREPROCESSED, DATASET_FOLDER_WEIGHTS, DATASET_SUB_FOLDER_VALIDATION, \
    TRAIN_MODEL_FILE, DATASET_CATEGORIES
from src.helper import log


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_name', type=str, required=True)
    parser.add_argument('--weights_folder', type=str, default=DATASET_FOLDER_WEIGHTS)
    parser.add_argument('--validation_h5', type=str,
                        default=os.path.join(DATASET_FOLDER_PREPROCESSED, f'{DATASET_SUB_FOLDER_VALIDATION}.h5'))
    parser.add_argument('--cut', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def main(weights_name, weights_folder, validation_h5, cut, verbose):
    model_path = os.path.join(weights_folder, weights_name, TRAIN_MODEL_FILE)
    assert os.path.isfile(model_path)

    x_train, y_train = loader.load(validation_h5, DATASET_SUB_FOLDER_VALIDATION, cut=cut)
    model = tf.keras.models.load_model(model_path)
    model.summary()
    loss, acc = model.evaluate(x_train, y_train, verbose=2)
    log.info('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

    main_category_achieved = list()
    for pred, gt in zip(model.predict(x_train), y_train):
        pred_idx = int(np.argmax(pred))
        label = list(DATASET_CATEGORIES.keys())[pred_idx]
        confidence = float(pred[pred_idx])
        gt_label = list(DATASET_CATEGORIES.keys())[int(np.argmax(gt))]
        achieved = label == gt_label
        if verbose:
            log.info('{}: {:5.2f}% -> {}{}'.format(label, 100 * confidence, gt_label, ' | YAY' if achieved else ''))

        main_category_achieved.append(
            DATASET_CATEGORIES[label]['main_category'] == DATASET_CATEGORIES[gt_label]['main_category']
        )

    log.info('Main category accuracy: {:5.2f}%'.format(100 * sum(main_category_achieved) / len(main_category_achieved)))


if __name__ == '__main__':
    args = parse_args()
    main(args.weights_name, args.weights_folder, args.validation_h5, args.cut, args.verbose)
