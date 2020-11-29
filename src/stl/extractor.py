import math
import cv2
import numpy as np

# noinspection PyPackageRequirements
from stl import mesh

from src.config import TRAIN_INPUT_SIZE
from src.helper import log


def extract_features(stl_file_list):
    try:
        thing_mesh = mesh.Mesh.from_files(stl_file_list, calculate_normals=False)
        thing_features = thing_mesh.points

        image_dimensions = int(math.sqrt(thing_features.size / 3))  # Compute ideal image dimensions
        amount_values = pow(image_dimensions, 2) * 3  # Compute needed values from STL array

        thing_features = thing_features.reshape(thing_features.size)  # Flatten audio array
        thing_features = np.random.choice(thing_features, size=amount_values)  # Get the needed amount values
        thing_features = thing_features.reshape(image_dimensions, image_dimensions, 3)  # Reshape to the image shape
        thing_features = (thing_features - thing_features.min()) * (1 / (thing_features.max() - thing_features.min()) * 255)
        thing_features = thing_features.astype('uint8')

        thing_features = cv2.resize(thing_features, (TRAIN_INPUT_SIZE, TRAIN_INPUT_SIZE), interpolation=cv2.INTER_LANCZOS4)
        thing_features = thing_features.flatten()
        return thing_features
    except Exception as e:
        log.debug(f'Error extracting features from STLs: [{e}]')
        return None
