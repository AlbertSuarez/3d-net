import os
import requests
import zipfile

from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm

from src.config import THINGIVERSE_API_NUMBER_PAGES, DATASET_CATEGORIES, THINGIVERSE_API_SEARCH, \
    THINGIVERSE_API_PACKAGE, DATASET_FOLDER_DOWNLOADED, THINGIVERSE_API_PER_PAGE, THINGIVERSE_API_CONCURRENCY, \
    THINGIVERSE_API_CONCURRENCY_DOWNLOAD, REQUEST_TIMEOUT
from src.helper import log, request


def _get_public_url(args):
    thing_id, params = args
    endpoint = THINGIVERSE_API_PACKAGE.format(thing_id)
    response = request.execute('GET', endpoint, params=params)
    return (response.get('public_url'), thing_id) if response else None


def _download(args):
    package_url, category_folder, category_id, thing_id = args
    try:
        zip_path = os.path.join(category_folder, f'{category_id}__{thing_id}.zip')
        if not os.path.isfile(zip_path) or not zipfile.is_zipfile(zip_path):
            response = requests.get(package_url, timeout=REQUEST_TIMEOUT)
            with open(zip_path, 'wb') as fd:
                fd.write(response.content)
            return True
    except Exception as e:
        log.debug(f'Error downloading thing {thing_id} | {package_url}: {e}')
    return False


def download_models(access_token):
    os.makedirs(DATASET_FOLDER_DOWNLOADED, exist_ok=True)
    category_items = DATASET_CATEGORIES.items()
    params = dict(access_token=access_token)
    for page_number in tqdm(range(THINGIVERSE_API_NUMBER_PAGES), total=THINGIVERSE_API_NUMBER_PAGES, desc='Pages'):
        for category_id, category_dict in tqdm(category_items, total=len(category_items), desc='Categories'):
            category_folder = os.path.join(DATASET_FOLDER_DOWNLOADED, category_id)
            endpoint = THINGIVERSE_API_SEARCH.format(
                page_number + 1, THINGIVERSE_API_PER_PAGE, category_dict.get('category_id')
            )
            response = request.execute('GET', endpoint, params=params)
            if response:
                thing_list = response.get('hits', [])
                os.makedirs(category_folder, exist_ok=True)
                if thing_list:
                    args = [(thing_item.get('id'), params) for thing_item in thing_list]
                    with ThreadPool(THINGIVERSE_API_CONCURRENCY) as pool:
                        public_url_list = [
                            i for i in
                            list(tqdm(
                                pool.imap(_get_public_url, args, chunksize=1), total=len(args), desc='URLs'
                            )) if i
                        ]
                    if public_url_list:
                        args = [(p, category_folder, category_id, i) for p, i in public_url_list]
                        with ThreadPool(THINGIVERSE_API_CONCURRENCY_DOWNLOAD) as pool:
                            # noinspection PyUnusedLocal
                            success = list(tqdm(
                                pool.imap(_download, args, chunksize=1), total=len(args), desc='Download'
                            ))
