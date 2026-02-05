from typing import Optional, Dict

import json
import os
from pathlib import Path

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


def open_and_extract_coco_json(
    gcp_cracc: CrAcc,
    coco_dataset_path: str
) -> Dict:

    mygcs: DaoCloudStorage = DaoCloudStorage(credential_accessor = gcp_cracc)

    # TODO: create json processor
    mygcs.process_blob_with_handler()