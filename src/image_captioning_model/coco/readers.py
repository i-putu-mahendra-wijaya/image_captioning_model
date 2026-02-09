from typing import List, Dict

import ijson

from google.cloud.storage.fileio import BlobReader

from src.image_captioning_model.GCP.CredentialAccessor import CredentialAccessor as CrAcc
from src.image_captioning_model.GCP.DaoCloudStorage.DaoCloudStorage import DaoCloudStorage
from src.image_captioning_model.coco.schemas import CocoBlob

def extract_image_component(
        blob_reader: BlobReader
) -> Dict[int, str]:

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
) -> Dict[int, List[str]]:

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
        coco_blob: CocoBlob
) -> Dict[int, List[str]]:

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
        coco_blob: Represents the GCS object locations required to read COCO metadata.

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

    image_dict: Dict[int, str] = mygcs.stream_blob_with_handler(
        bucket_name = coco_blob.bucket_name,
        object_name = coco_blob.image_blob,
        callback_func = extract_image_component
    )

    anno_dict: Dict[int, List[str]] = mygcs.stream_blob_with_handler(
        bucket_name = coco_blob.bucket_name,
        object_name = coco_blob.caption_blob,
        callback_func = extract_annotation_component
    )

    combined_dict: Dict[int, List[str]] = {}

    for each_image_id in image_dict.keys():
        file_name: str = image_dict.get(each_image_id, "")
        annotations: List[str] = anno_dict.get(each_image_id, [""])
        combined_dict[each_image_id] = [file_name]  + annotations

    return combined_dict