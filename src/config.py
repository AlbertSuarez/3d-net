import os


THINGIVERSE_FLASK_PORT = 8080
THINGIVERSE_FLASK_WAIT_PRE = 3
THINGIVERSE_FLASK_WAIT_POST = 300
THINGIVERSE_FLASK_WAIT_ENABLE = False
THINGIVERSE_FLASK_ENDPOINT = '/download'
THINGIVERSE_API_NUMBER_PAGES = 1000
THINGIVERSE_API_PER_PAGE = 500
THINGIVERSE_API_CONCURRENCY = 10
THINGIVERSE_API_CONCURRENCY_DOWNLOAD = 50
THINGIVERSE_API_AUTH = 'https://www.thingiverse.com/login/oauth/authorize'
THINGIVERSE_API_TOKEN = 'https://www.thingiverse.com/login/oauth/access_token'
THINGIVERSE_API_DONE = 'https://asuarez.dev/3d-net/docs/images/done.jpeg'
THINGIVERSE_API_PACKAGE = 'https://api.thingiverse.com/things/{}/package-url'
THINGIVERSE_API_SEARCH = 'https://api.thingiverse.com/search/' \
                         '?page={}&per_page={}&sort=popular&category_id={}&type=things'

DATASET_FOLDER = 'data'
DATASET_FOLDER_DOWNLOADED = os.path.join(DATASET_FOLDER, 'downloaded')
DATASET_FOLDER_STANDARDIZED = os.path.join(DATASET_FOLDER, 'standardized')
DATASET_FOLDER_PREPROCESSED = os.path.join(DATASET_FOLDER, 'preprocessed')
DATASET_FOLDER_WEIGHTS = os.path.join(DATASET_FOLDER, 'weights')
DATASET_FOLDER_WEIGHTS_FINAL = 'weights'
DATASET_SUB_FOLDER_TRAINING = 'training'
DATASET_SUB_FOLDER_VALIDATION = 'validation'
DATASET_CATEGORIES = {
    '3d__printer_accessories': {'category_id': 127, 'main_category': '3d'},
    '3d__printer_extruders': {'category_id': 152, 'main_category': '3d'},
    '3d__printer_parts': {'category_id': 128, 'main_category': '3d'},
    '3d__printers': {'category_id': 126, 'main_category': '3d'},
    '3d__printing_tests': {'category_id': 129, 'main_category': '3d'},
    'art__2d': {'category_id': 144, 'main_category': 'art'},
    'art__tools': {'category_id': 75, 'main_category': 'art'},
    'art__coins_badges': {'category_id': 143, 'main_category': 'art'},
    'art__interactive': {'category_id': 78, 'main_category': 'art'},
    'art__math': {'category_id': 79, 'main_category': 'art'},
    'art__scans_replicas': {'category_id': 145, 'main_category': 'art'},
    'art__sculptures': {'category_id': 80, 'main_category': 'art'},
    'art__signs_logos': {'category_id': 76, 'main_category': 'art'},
    'fashion__accessories': {'category_id': 81, 'main_category': 'fashion'},
    'fashion__bracelets': {'category_id': 82, 'main_category': 'fashion'},
    'fashion__costume': {'category_id': 142, 'main_category': 'fashion'},
    'fashion__earrings': {'category_id': 139, 'main_category': 'fashion'},
    'fashion__glasses': {'category_id': 83, 'main_category': 'fashion'},
    'fashion__jewelry': {'category_id': 84, 'main_category': 'fashion'},
    'fashion__keychains': {'category_id': 130, 'main_category': 'fashion'},
    'fashion__rings': {'category_id': 85, 'main_category': 'fashion'},
    'gadgets__audio': {'category_id': 141, 'main_category': 'gadgets'},
    'gadgets__camera': {'category_id': 86, 'main_category': 'gadgets'},
    'gadgets__computer': {'category_id': 87, 'main_category': 'gadgets'},
    'gadgets__mobile_phone': {'category_id': 88, 'main_category': 'gadgets'},
    'gadgets__tablet': {'category_id': 90, 'main_category': 'gadgets'},
    'gadgets__video_games': {'category_id': 91, 'main_category': 'gadgets'},
    'hobby__automotive': {'category_id': 155, 'main_category': 'hobby'},
    'hobby__diy': {'category_id': 93, 'main_category': 'hobby'},
    'hobby__electronics': {'category_id': 92, 'main_category': 'hobby'},
    'hobby__music': {'category_id': 94, 'main_category': 'hobby'},
    'hobby__rc_vehicles': {'category_id': 95, 'main_category': 'hobby'},
    'hobby__robotics': {'category_id': 96, 'main_category': 'hobby'},
    'hobby__sport_outdoors': {'category_id': 140, 'main_category': 'hobby'},
    'household__bathroom': {'category_id': 147, 'main_category': 'household'},
    'household__containers': {'category_id': 146, 'main_category': 'household'},
    'household__decor': {'category_id': 97, 'main_category': 'household'},
    'household__supplies': {'category_id': 99, 'main_category': 'household'},
    'household__kitchen_dining': {'category_id': 100, 'main_category': 'household'},
    'household__office_organization': {'category_id': 101, 'main_category': 'household'},
    'household__outdoor_garden': {'category_id': 98, 'main_category': 'household'},
    'household__pets': {'category_id': 103, 'main_category': 'household'},
    'learning__biology': {'category_id': 106, 'main_category': 'learning'},
    'learning__engineering': {'category_id': 104, 'main_category': 'learning'},
    'learning__math': {'category_id': 105, 'main_category': 'learning'},
    'learning__physics_astronomy': {'category_id': 148, 'main_category': 'learning'},
    'models__animals': {'category_id': 107, 'main_category': 'models'},
    'models__buildings_structures': {'category_id': 108, 'main_category': 'models'},
    'models__creatures': {'category_id': 109, 'main_category': 'models'},
    'models__food_drink': {'category_id': 110, 'main_category': 'models'},
    'models__furniture': {'category_id': 111, 'main_category': 'models'},
    'models__robots': {'category_id': 115, 'main_category': 'models'},
    'models__people': {'category_id': 112, 'main_category': 'models'},
    'models__props': {'category_id': 114, 'main_category': 'models'},
    'models__vehicles': {'category_id': 116, 'main_category': 'models'},
    'tools__hand': {'category_id': 118, 'main_category': 'tools'},
    'tools__machine': {'category_id': 117, 'main_category': 'tools'},
    'tools__holders_boxes': {'category_id': 120, 'main_category': 'tools'},
    'toys__chess': {'category_id': 151, 'main_category': 'toys'},
    'toys__construction': {'category_id': 121, 'main_category': 'toys'},
    'toys__dice': {'category_id': 122, 'main_category': 'toys'},
    'toys__games': {'category_id': 123, 'main_category': 'toys'},
    'toys__mechanical': {'category_id': 124, 'main_category': 'toys'},
    'toys__playsets': {'category_id': 113, 'main_category': 'toys'},
    'toys__puzzles': {'category_id': 125, 'main_category': 'toys'},
    'toys__accessories': {'category_id': 149, 'main_category': 'toys'}
}

REQUEST_TIMEOUT = 30

STANDARDIZE_CONCURRENCY = 30

TRAIN_EPOCHS = len(DATASET_CATEGORIES)
TRAIN_INPUT_SIZE = 256
TRAIN_STRATEGY = 'sigmoid'  # (relu, sigmoid)
TRAIN_MODEL_FILE = 'model.h5'
