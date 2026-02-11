from typing import Dict, List, Optional

from pathlib import Path
import tempfile
import gzip
import json

from keras import Model
from keras.src.applications import VGG19

import numpy as np

from google.cloud.storage import Blob

from src.image_captioning_model.GCP.DaoCloudStorage import DaoCloudStorage

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

    vgg19: Model = VGG19()

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


def save_caption_dict_to_gcs(
        mygcs: DaoCloudStorage,
        bucket_name: str,
        object_name: str,
        caption_dict: Dict [int, List[str]]
) -> None:
    """
    Serialize a caption dictionary and upload it to Google Cloud Storage (GCS).

    This function writes the provided ``caption_dict`` into a temporary file
    as JSON content, compresses it using gzip, and uploads it to the specified
    GCS bucket using the provided ``DaoCloudStorage`` instance.

    If ``object_name`` does not end with ``.jsonl``, the extension will be
    automatically appended before upload.

    The upload process is performed using a temporary directory to avoid
    persisting intermediate files on disk.

    :param mygcs:
        Cloud Storage DAO responsible for handling upload operations.
    :type mygcs: DaoCloudStorage

    :param bucket_name:
        Target GCS bucket name.
    :type bucket_name: str

    :param object_name:
        Destination object name in GCS. The ``.jsonl`` extension will be added
        automatically if not present.
    :type object_name: str

    :param caption_dict:
        Mapping from image ID to a list of captions associated with that image.
    :type caption_dict: Dict[int, List[str]]

    :raises RuntimeError:
        If the upload to GCS fails.

    :return:
        None
    :rtype: None
    """

    if not object_name.endswith(".jsonl"):
        object_name = object_name + ".jsonl"

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path: Path =  Path(tmpdir) / Path(object_name).name

        with gzip.open(local_path, "wt", encoding="utf-8") as fhandler:
            fhandler.write(json.dumps(caption_dict))

        uploaded_blob: Optional[Blob] = mygcs.upload_blob_from_file(
            bucket_name = bucket_name,
            object_name = object_name,
            file_path = local_path
        )

        if uploaded_blob is None:
            raise RuntimeError(f"Filed to upload prediction to gs://{bucket_name}/{object_name}")


def save_prediction_npz_to_gcs(
    mygcs: DaoCloudStorage,
    bucket_name: str,
    object_name: str,
    prediction: np.ndarray,
    array_key: str = "pred"
) -> None:

    """
    Serialize a NumPy prediction array to a temporary NPZ file and upload it to GCS.

    Args:
        mygcs: DaoCloudStorage instance.
        bucket_name: Target GCS bucket.
        object_name: Target GCS object path (should end with ".npz").
        prediction: NumPy array to serialize.
        array_key: Key name used inside the NPZ container.
    """

    if not object_name.endswith(".npz"):
        object_name = object_name + ".npz"

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path: Path = Path(tmpdir) / Path(object_name).name
        np.savez_compressed(local_path, **{array_key: prediction})

        upload_blob = mygcs.upload_blob_from_file(
            bucket_name = bucket_name,
            object_name = object_name,
            file_path = local_path
        )

        if upload_blob is None:
            raise RuntimeError(f"Filed to upload prediction to gs://{bucket_name}/{object_name}")