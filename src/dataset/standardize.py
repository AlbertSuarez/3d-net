import argparse
import glob
import os
import random
import shutil
import zipfile

from tqdm import tqdm

from src.config import DATASET_FOLDER_DOWNLOADED, DATASET_CATEGORIES, DATASET_FOLDER_STANDARDIZED, \
    DATASET_SUB_FOLDER_TRAINING, DATASET_SUB_FOLDER_VALIDATION
from src.helper import log


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--downloaded_folder', type=str, default=DATASET_FOLDER_DOWNLOADED)
    parser.add_argument('--standardized_folder', type=str, default=DATASET_FOLDER_STANDARDIZED)
    parser.add_argument('--training_factor', type=float, default=0.95)
    parser.add_argument('--max_folders', type=int, default=0)
    return parser.parse_args()


def _validate_input(downloaded_folder, training_factor):
    if not os.path.isdir(downloaded_folder) or len(glob.glob(f'{downloaded_folder}/*/')) != len(DATASET_CATEGORIES):
        raise ValueError('Input folder not valid.')
    if not (0.0 <= training_factor <= 0.95):
        raise ValueError('Invalid training factor.')


def _validate_output(standardized_folder):
    if os.path.isdir(standardized_folder):
        shutil.rmtree(standardized_folder)
    training_folder = os.path.join(standardized_folder, DATASET_SUB_FOLDER_TRAINING)
    validation_folder = os.path.join(standardized_folder, DATASET_SUB_FOLDER_VALIDATION)
    os.makedirs(training_folder)
    os.makedirs(validation_folder)
    return training_folder, validation_folder


def _standardize(downloaded_folder, training_folder, validation_folder, training_factor, max_folders):
    valid_extensions = ('.stl', '.STL')
    folder_list = os.listdir(downloaded_folder)
    for category_folder in tqdm(folder_list, total=len(folder_list), desc='Categories'):
        os.makedirs(os.path.join(training_folder, category_folder))
        os.makedirs(os.path.join(validation_folder, category_folder))
        downloaded_category = os.path.join(downloaded_folder, category_folder)
        if os.path.isdir(downloaded_category):
            category_list = os.listdir(downloaded_category)
            if max_folders > 0:
                category_list = category_list[:max_folders]
            for zip_path in tqdm(category_list, total=len(category_list), desc='Files'):
                try:
                    with zipfile.ZipFile(os.path.join(downloaded_category, zip_path)) as zip_file:
                        for zip_info in zip_file.infolist():
                            if zip_info.filename.endswith(valid_extensions):
                                zip_info.filename = os.path.basename(zip_info.filename).lower()
                                zip_file.extract(
                                    zip_info, os.path.join(
                                        training_folder, category_folder,
                                        os.path.splitext(
                                            os.path.basename(os.path.join(downloaded_category, zip_path))
                                        )[0]
                                    )
                                )
                except Exception as e:
                    log.debug(f'Error extracting {zip_path}: [{e}]')
            all_items_folder = os.path.join(training_folder, category_folder)
            all_items = [
                os.path.join(all_items_folder, d) for d in os.listdir(all_items_folder)
                if os.path.isdir(os.path.join(all_items_folder, d))
            ]
            for f in random.sample(all_items, int(len(all_items) * (1 - training_factor))):
                shutil.move(f, os.path.join(validation_folder, category_folder, os.path.basename(f)))


def _count(training_folder, validation_folder):
    it = [(DATASET_SUB_FOLDER_TRAINING, training_folder), (DATASET_SUB_FOLDER_VALIDATION, validation_folder)]
    for name, path in it:
        total = 0
        log.info(name.upper())
        for category_name in os.listdir(path):
            category_folder = os.path.join(path, category_name)
            if os.path.isdir(category_folder):
                category_count = len(glob.glob(f'{category_folder}/*/*'))
                total += category_count
                log.info('{}: {} examples'.format(category_name, category_count))
        log.info('total: {} examples'.format(total))


def main(downloaded_folder, standardized_folder, training_factor, max_folders):
    _validate_input(downloaded_folder, training_factor)
    training_folder, validation_folder = _validate_output(standardized_folder)
    _standardize(downloaded_folder, training_folder, validation_folder, training_factor, max_folders)
    _count(training_folder, validation_folder)


if __name__ == '__main__':
    args = parse_args()
    main(args.downloaded_folder, args.standardized_folder, args.training_factor, args.max_folders)
