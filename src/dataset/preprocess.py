import argparse
import os
import h5py
import numpy as np

from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm

from src.config import DATASET_FOLDER_STANDARDIZED, DATASET_SUB_FOLDER_TRAINING, DATASET_FOLDER_PREPROCESSED, \
    DATASET_SUB_FOLDER_VALIDATION, STANDARDIZE_CONCURRENCY, DATASET_CATEGORIES
from src.helper import log
from src.stl import extractor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed_folder', type=str, default=DATASET_FOLDER_PREPROCESSED)
    parser.add_argument('--training_data', type=str,
                        default=os.path.join(DATASET_FOLDER_STANDARDIZED, DATASET_SUB_FOLDER_TRAINING))
    parser.add_argument('--validation_data', type=str,
                        default=os.path.join(DATASET_FOLDER_STANDARDIZED, DATASET_SUB_FOLDER_VALIDATION))
    return parser.parse_args()


def main(preprocessed_folder, training_data, validation_data):
    os.makedirs(preprocessed_folder, exist_ok=True)
    t_folder_list = [d for d in os.listdir(training_data) if os.path.isdir(os.path.join(training_data, d))]
    v_folder_list = [d for d in os.listdir(validation_data) if os.path.isdir(os.path.join(validation_data, d))]
    for name, path in [(DATASET_SUB_FOLDER_TRAINING, t_folder_list), (DATASET_SUB_FOLDER_VALIDATION, v_folder_list)]:
        log.info(name)
        x_data = list()
        y_data = list()
        for category_name in tqdm(path, total=len(path)):
            stl_array = list()
            for thing_folder in os.listdir(os.path.join(training_data, category_name)):
                thing_folder = os.path.join(training_data, category_name, thing_folder)
                stl_array.append([os.path.join(thing_folder, s) for s in os.listdir(thing_folder)])
            with ThreadPool(STANDARDIZE_CONCURRENCY) as pool:
                category_things = list(tqdm(
                    pool.imap(extractor.extract_features, stl_array, chunksize=1), total=len(stl_array)
                ))
                category_things = [c for c in category_things if c is not None]
            x_data.extend(category_things)
            y_data.extend([list(DATASET_CATEGORIES.keys()).index(category_name)] * len(category_things))

        x_data = np.array(x_data)
        y_data = np.array(y_data)
        with h5py.File(os.path.join(preprocessed_folder, f'{name}.h5'), 'w') as reduced_features_file:
            reduced_features_file.create_dataset('x_train', data=x_data)
            reduced_features_file.create_dataset('y_train', data=y_data)


if __name__ == '__main__':
    args = parse_args()
    main(args.preprocessed_folder, args.training_data, args.validation_data)
