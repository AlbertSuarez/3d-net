import os
import requests

from tqdm import tqdm

from src.config import THINGIVERSE_API_NUMBER_PAGES, DATASET_CATEGORIES, THINGIVERSE_API_SEARCH, \
    THINGIVERSE_API_PACKAGE, DATASET_FOLDER_DOWNLOADED, THINGIVERSE_API_PER_PAGE, REQUEST_TIMEOUT
from src.helper import log, request


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
                for thing_item in tqdm(thing_list, total=len(thing_list), desc='Files'):
                    thing_id = thing_item.get('id')
                    endpoint = THINGIVERSE_API_PACKAGE.format(thing_id)
                    response = request.execute('GET', endpoint, params=params)
                    if response:
                        package_url = response.get('public_url')
                        if package_url:
                            try:
                                zip_path = os.path.join(category_folder, f'{category_id}__{thing_id}.zip')
                                if not os.path.isfile(zip_path):
                                    response = requests.get(package_url, timeout=REQUEST_TIMEOUT)
                                    with open(zip_path, 'wb') as fd:
                                        fd.write(response.content)
                            except Exception as e:
                                log.warn(f'Error downloading thing {thing_id} | {package_url}: {e}')
