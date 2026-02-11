from typing import Dict, List

from pathlib import Path
import tempfile

from keras import Model
from keras.src.applications import VGG19

import numpy as np

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
    # TODO: implement save caption dict to gcs
    pass


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