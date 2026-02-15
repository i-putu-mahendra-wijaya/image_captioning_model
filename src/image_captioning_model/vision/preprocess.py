import io
from typing import Tuple

import numpy as np
from PIL import Image
from PIL.ImageFile import ImageFile
from google.cloud.storage.fileio import BlobReader

from image_captioning_model.GCP.CredentialAccessor.CredentialAccessor import CredentialAccessor as CrAcc
from image_captioning_model.GCP.DaoCloudStorage.DaoCloudStorage import DaoCloudStorage


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
