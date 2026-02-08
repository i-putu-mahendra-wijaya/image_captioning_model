from typing import Optional, Dict, List, Tuple

from dataclasses import dataclass, asdict
import os
import io
from pathlib import Path
from pprint import pprint

import numpy as np
import ijson
from PIL.ImageFile import ImageFile, Image

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

COCO_BUCKET_NAME: str = os.getenv("COCO_BUCKET_NAME", "COCO_BUCKET_NAME not found")
COCO_TRAIN_IMAGE_FOLDER: str = os.getenv("COCO_TRAIN_IMAGE_FOLDER", "COCO_TRAIN_IMAGE_FOLDER not found")
COCO_TRAIN_LABEL_METADATA: str = os.getenv("COCO_TRAIN_LABEL_METADATA", "COCO_TRAIN_LABEL_METADATA not found")
COCO_TRAIN_CAPTION_METADATA:str = os.getenv("COCO_TRAIN_CAPTION_METADATA", "COCO_TRAIN_CAPTION_METADATA not found")

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


@dataclass
class CocoBlob:
    """Represents the GCS object locations required to read COCO metadata.

    Attributes:
        bucket_name: Name of the Google Cloud Storage bucket that contains COCO files.
        image_blob: Object name (path) to the COCO JSON containing the `images` array.
        caption_blob: Object name (path) to the COCO JSON containing the `annotations` array.
    """

    bucket_name: str
    image_blob: str
    caption_blob: str


coco_blob: CocoBlob = CocoBlob(
    bucket_name = COCO_BUCKET_NAME,
    image_blob  = COCO_TRAIN_CAPTION_METADATA,
    caption_blob = COCO_TRAIN_CAPTION_METADATA,
)

def extract_image_component(
    blob_reader: BlobReader
) -> Dict:

    """Extract a mapping of COCO image IDs to image file names from a COCO JSON stream.

    This function expects the underlying JSON to contain an `images` array, where each
    element is a dict containing at least:
      - `id`: the integer image ID
      - `file_name`: the corresponding image filename

    The parsing is performed in a streaming manner using `ijson.items`, so the entire
    JSON does not need to be loaded into memory.

    Args:
        blob_reader: A file-like stream (e.g., GCS BlobReader) positioned at the start
            of a COCO-format JSON file.

    Returns:
        A dictionary mapping image_id -> file_name.
        Example: {391895: "COCO_train2014_000000391895.jpg", ...}
    """

    image_dict: Dict = {}

    print("Processing images ...")
    images = ijson.items(blob_reader, "images.item")

    # 1. Processing 'images' array item-by-item
    # The 'images.item' prefix tells ijson to get 'images' key
    # and yield each dictionary inside that list
    for each_image in images:
        image_dict[each_image["id"]] = each_image["file_name"]

    return image_dict


def extract_annotation_component(
    blob_reader: BlobReader
) -> Dict:

    """Extract a mapping of COCO image IDs to caption strings from a COCO JSON stream.

    This function expects the underlying JSON to contain an `annotations` array, where
    each element is a dict containing at least:
      - `image_id`: the integer image ID that the caption belongs to
      - `caption`: the caption text

    Captions are grouped by `image_id`, producing a dict where each key maps to a list
    of captions for that image.

    Args:
        blob_reader: A file-like stream (e.g., GCS BlobReader) positioned at the start
            of a COCO-format JSON file.

    Returns:
        A dictionary mapping image_id -> list of captions.
        Example: {391895: ["A man riding a bike ...", "A person on ..."], ...}
    """

    anno_dict: Dict = {}

    print("Processing annotations ...")
    annos = ijson.items(blob_reader, "annotations.item")

    # 1. Processing 'annotations' array item-by-item
    # The 'annotations.item' prefix tells ijson to get 'images' key
    # and yield each dictionary inside that list
    for each_anno in annos:
        if each_anno["image_id"] not in anno_dict.keys():
            anno_dict[each_anno["image_id"]] = [each_anno["caption"]]
        else:
            anno_dict[each_anno["image_id"]].append(each_anno["caption"])

    return anno_dict

def open_and_extract_coco_json(
    gcp_cracc: CrAcc,
) -> Dict:

    """Read COCO metadata JSON from GCS and combine file names with captions per image.

    This function:
      1) Instantiates a `DaoCloudStorage` using the provided credential accessor.
      2) Streams the COCO JSON file(s) from GCS.
      3) Extracts:
         - image_id -> file_name
         - image_id -> [caption_1, caption_2, ...]
      4) Combines them into a single dictionary:
         image_id -> [file_name, caption_1, caption_2, ...]

    Notes:
        - This implementation assumes the provided `DaoCloudStorage.stream_blob_with_handler`
          resets/opens the stream for each call. The `extract_*` functions are designed to
          parse from the start of the JSON stream.
        - Missing keys are handled defensively via `.get()`. If an image has no captions,
          it will receive a default list of `[""]` (current behavior).

    Args:
        gcp_cracc: Credential accessor used to authenticate to GCS.

    Returns:
        A dictionary mapping image_id -> list containing the file name followed by
        one or more captions.

        Example:
            {
                391895: [
                    "COCO_train2014_000000391895.jpg",
                    "A man riding a bike down the street.",
                    "A bicyclist rides in traffic."
                ],
                ...
            }
    """

    mygcs: DaoCloudStorage = DaoCloudStorage(credential_accessor = gcp_cracc)

    image_dict: Dict = mygcs.stream_blob_with_handler(
        bucket_name = coco_blob.bucket_name,
        object_name = coco_blob.image_blob,
        callback_func = extract_image_component
    )

    anno_dict: Dict = mygcs.stream_blob_with_handler(
        bucket_name = coco_blob.bucket_name,
        object_name = coco_blob.caption_blob,
        callback_func = extract_annotation_component
    )

    combined_dict: Dict = {}

    for each_image_id in image_dict.keys():
        file_name: str = image_dict.get(each_image_id, "")
        annotations: List[str] = anno_dict.get(each_image_id, [""])
        combined_dict[each_image_id] = [file_name]  + annotations

    return combined_dict


def check_malformed_entries(
        combined_dict: Dict
) -> Tuple:

    """Validate combined COCO metadata dictionary for malformed entries.

    This function scans the dictionary produced by `open_and_extract_coco_json`
    and detects two potential data integrity issues:

    1. Empty-string image IDs ("")
    2. Image IDs that do not have captions

    The expected structure of `combined_dict` is:

        image_id -> [file_name, caption_1, caption_2, ...]

    Args:
        combined_dict: Dictionary mapping image IDs to a list containing the
            file name followed by one or more captions.

    Returns:
        A tuple containing:
            - empty_string_keys: list of image IDs equal to ""
            - images_without_captions: list of image IDs whose value list
              contains only the file name (length <= 1)
    """

    empty_string_keys: List[str] = []
    images_without_captions: List[str] = []

    for each_image_id, values in combined_dict.items():

        if each_image_id == "":
            empty_string_keys.append(each_image_id)

        if len(values) <= 1:
            images_without_captions.append(each_image_id)

    return empty_string_keys, images_without_captions


def create_base_model(

) -> Model:

    """Create a VGG19 feature-extraction model for transfer learning.

    This function loads the pretrained VGG19 model (ImageNet weights) and
    constructs a truncated model that outputs the feature map from the
    `block5_conv4` layer. The resulting model can be used as a fixed
    convolutional feature extractor for image captioning or other
    vision tasks.

    Returns:
        A TensorFlow Keras `Model` whose output corresponds to the
        `block5_conv4` layer of VGG19.
    """

    vgg19: Model = VGG19(weights="imagenet")

    print("VGG19 Model Summary ...")
    vgg19.summary()

    print("Create base model from VGG19 as the base model for Transfer Learning later ... ")
    base_model: Model = Model(
        inputs = vgg19.input,
        outputs = vgg19.get_layer("block5_conv4").output
    )

    print("Base Model Summary ...")
    base_model.summary()

    return base_model


def resize_and_crop_image(
    blob_reader: BlobReader,
) -> np.ndarray:

    """Resize and center-crop an image to VGG-compatible input size.

    This function:
    1. Loads an image from a file-like stream.
    2. Resizes the image so that the shortest side becomes 256 pixels.
    3. Performs a center crop to 224x224 pixels.
    4. Converts the image to a NumPy array.
    5. Adds a batch dimension (shape becomes `(1, 224, 224, 3)`).

    This preprocessing matches the typical input preparation used for
    VGG-style convolutional networks.

    Args:
        blob_reader: A file-like object (e.g., GCS BlobReader) containing
            image data.

    Returns:
        A NumPy array of shape `(1, 224, 224, 3)` representing the
        resized and cropped image.
    """

    # Determine dimensions
    img_bytes: bytes = blob_reader.read()
    blob_reader.seek(0)

    image: ImageFile = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    width: int = image.size[0]
    height: int = image.size[1]

    # Resize so the shortest side is 256 pixels
    if height > width:
        new_size: Tuple[int, int] = (
            256,
            int(height / width * 256)
        )
    else:
        new_size: Tuple[int, int] = (
            int(width / height * 256),
            256
        )

    resized_image = image.resize(new_size)

    resized_width: int = resized_image.size[0]
    resized_height: int = resized_image.size[1]

    resized_image_np: np.ndarray = np.array(resized_image)

    # Crop to center 224 x 224 region
    h_start: int = int((resized_height - 224) / 2)
    w_start: int = int((resized_width - 224) / 2)

    rc_image_np: np.ndarray = resized_image_np[
        h_start : h_start + 224,
        w_start : w_start + 224
    ]

    # Rearrange array to have one more dimension representing batch size = 1
    rc_image_np = np.expand_dims(
        rc_image_np,
        axis = 0
    )


    return rc_image_np


def preprocess_coco_image(
    gcp_cracc: CrAcc,
    bucket_name: str,
    image_folder_path: str,
    image_file_name: str
) -> np.ndarray:

    """Load and preprocess a COCO image from GCS for VGG19 input.

    COCO images have varying dimensions, while VGG19 expects fixed-size
    inputs. This function streams an image from Google Cloud Storage and
    applies resizing and center-cropping via `resize_and_crop_image`.

    Args:
        gcp_cracc: Credential accessor used for GCS authentication.
        bucket_name: GCS bucket containing the image.
        image_folder_path: Folder path inside the bucket.
        image_file_name: Name of the image file.

    Returns:
        A NumPy array of shape `(1, 224, 224, 3)` ready for model input.
    """

    mygcs: DaoCloudStorage = DaoCloudStorage(
        credential_accessor = gcp_cracc
    )

    rc_image_np: np.ndarray = mygcs.stream_blob_with_handler(
        bucket_name = bucket_name,
        object_name = (
            image_folder_path +
            image_file_name
        ),
        callback_func = resize_and_crop_image
    )

    return rc_image_np


if __name__ == "__main__":

    combined_dict: Dict = open_and_extract_coco_json(
        gcp_cracc = cracc
    )

    pprint(
        combined_dict,
        indent = 4
    )

    # Checking if there are any malformed entries
    empty_string_keys, images_without_captions = check_malformed_entries(
        combined_dict = combined_dict
    )

    print(f"Empty string keys: {len(empty_string_keys)}")
    print(f"Images without captions: {len(images_without_captions)}")

    for each_idx, each_image_id in enumerate(combined_dict.keys()):

        if each_idx % 1_000 == 0:
            print(f"Progress : {each_idx} images processed")

        list_val: List[str] = combined_dict[each_image_id]
        image_file_name: str = list_val[0]
        image_folder_path: str = COCO_TRAIN_IMAGE_FOLDER

        rc_image_np: np.ndarray = preprocess_coco_image(
            gcp_cracc = cracc,
            bucket_name = coco_blob.bucket_name,
            image_folder_path = image_folder_path,
            image_file_name = image_file_name
        )

        print(f"resized_cropped_image {image_file_name} size {rc_image_np.shape}")