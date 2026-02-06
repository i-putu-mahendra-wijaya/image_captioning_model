from typing import Optional, Dict

import ijson
import os
from pathlib import Path
from pprint import pprint

import numpy as np

# Suppress warnings from TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # 0=all, 1=INFO off, 2=INFO+WARN off, 3=INFO+WARN+ERROR off (best effort)
os.environ["AUTOGRAPH_VERBOSITY"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # optional: reduces some CPU info logs

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from google.cloud.storage.fileio import BlobReader

import pickle
import gzip
import logging
tf.get_logger().setLevel(logging.ERROR)

from dotenv import load_dotenv
load_dotenv()

from yaml import safe_load

from src.image_captioning_model.GCP.CredentialAccessor.CredentialAccessor import CredentialAccessor as CrAcc
from src.image_captioning_model.GCP.DaoCloudStorage.DaoCloudStorage import DaoCloudStorage


GCS_COCO_DATASET: str = os.getenv("GCS_COCO_DATASET", "GCS_COCO_DATASET not found")
GCS_OUTPUT_DIR: str = os.getenv("GCS_OUTPUT_DIR", "GCS_OUTPUT_DIR not found")
GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "GCP_PROJECT_ID not found")
GCP_PROJECT_LOCATION: str = os.getenv("GCP_PROJECT_LOCATION", "GCP_PROJECT_LOCATION not found")

cwd: Path = Path.cwd()
config_path: Path = cwd / "config.yaml"

with open(config_path, "r") as config_file:
    _config_content: Dict = safe_load(config_file)

PROJECT_HOME_PATH: Path = Path(_config_content["PROJECT_HOME_PATH"])
GCP_SA_PATH: Path = PROJECT_HOME_PATH / "credentials" / "gcp" / "service_account.json"

cracc: CrAcc = CrAcc(
    credential_path = GCP_SA_PATH,
    project_id = GCP_PROJECT_ID,
    location = GCP_PROJECT_LOCATION,
)

def extract_image_and_annotation_component(
    blob_reader: BlobReader
) -> Dict:

    """
    Extract image filenames and captions from a COCO-style JSON stream.

    This function incrementally parses a JSON document containing ``images``
    and ``annotations`` arrays (such as the COCO caption dataset format)
    using the :mod:`ijson` streaming parser. It builds a dictionary keyed
    by image ID, where each value contains the image filename followed by
    its associated captions.

    The function reads the stream twice:
    first to collect image metadata, and then to attach captions from the
    annotations section. The reader is rewound between passes using
    ``seek(0)``.

    Parameters
    ----------
    blob_reader : BlobReader
        A binary file-like stream opened from a GCS blob (for example via
        ``blob.open("rb")``). The stream must support ``seek()`` so that
        the JSON document can be parsed multiple times.

    Returns
    -------
    dict
        A dictionary mapping image IDs to a list containing the image
        filename followed by its captions.

        Structure example::

            {
                391895: [
                    "COCO_val2014_000000391895.jpg",
                    "A man riding a motorcycle on a dirt road.",
                    "A person on a motorbike in a rural area."
                ]
            }

    Notes
    -----
    - The JSON structure is expected to contain top-level keys:
      ``images`` and ``annotations``.
    - Each item in ``images`` must contain:
      ``id`` and ``file_name``.
    - Each item in ``annotations`` must contain:
      ``image_id`` and ``caption``.
    - The function uses streaming parsing via :mod:`ijson` to avoid loading
      the entire JSON document into memory.
    - The reader position is reset using ``blob_reader.seek(0)`` before
      processing annotations.

    Examples
    --------
    Use with :meth:`DaoCloudStorage.stream_blob_with_handler`:

    >>> dao.stream_blob_with_handler(
    ...     bucket_name="coco-dataset",
    ...     object_name="annotations/captions_val2017.json",
    ...     callback_func=extract_image_and_annotation_component
    ... )
    """

    image_dict: Dict = {}

    print("Processing images ...")
    images = ijson.items(blob_reader, "images.item")

    # 1. Processing 'images' array item-by-item
    # The 'images.item' prefix tells ijson to get 'images' key
    # and yield each dictionary inside that list
    for each_image in images:
        image_dict[each_image["id"]] = [each_image["file_name"]]

    # 2. Seek back to the start to process annotations
    blob_reader.seek(0)

    # 3. Process 'annotations' array item-by-item
    print("Processing annotations ... ")
    annotations = ijson.items(blob_reader, "annotations.item")

    for each_annotation in annotations:
        image_id = each_annotation["image_id"]
        if image_id in image_dict:
            image_dict[image_id].append(each_annotation["caption"])

    print(f"Finished processing {len(image_dict)} images ...")
    return image_dict

def open_and_extract_coco_json(
    gcp_cracc: CrAcc,
) -> Dict:

    mygcs: DaoCloudStorage = DaoCloudStorage(credential_accessor = gcp_cracc)

    image_dict: Dict = mygcs.stream_blob_with_handler(
        bucket_name = "dsp_coco_dataset",
        object_name = "train/labels.json",
        callback_func = extract_image_and_annotation_component
    )

    return image_dict


if __name__ == "__main__":

    image_dict: Dict = open_and_extract_coco_json(
        gcp_cracc = cracc
    )

    pprint(
        image_dict,
        indent = 4
    )