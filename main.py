from typing import Dict, List

from pathlib import Path
from pprint import pprint

import numpy as np

from src.image_captioning_model.settings.env import EnvVar, get_env
from src.image_captioning_model.settings.config import ProjectConfig, load_project_config

env_var: EnvVar = get_env()

project_root_path: Path = Path.cwd()
project_config: ProjectConfig = load_project_config(
    project_root_path = project_root_path,
)

from src.image_captioning_model.vision.preprocess import preprocess_coco_image
from src.image_captioning_model.coco.schemas import CocoBlob
from src.image_captioning_model.coco.readers import open_and_extract_coco_json
from src.image_captioning_model.coco.validators import check_malformed_entries
from src.image_captioning_model.GCP.CredentialAccessor.CredentialAccessor import CredentialAccessor as CrAcc

import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

def main(

) -> None:
    cracc: CrAcc = CrAcc(
        credential_path = project_config.gcp_sa_path,
        project_id = env_var.gcp_project_id,
        location = env_var.gcp_project_location,
    )


    coco_blob: CocoBlob = CocoBlob(
        bucket_name = env_var.coco_bucket_name,
        image_blob  = env_var.coco_train_label_metadata,
        caption_blob = env_var.coco_train_caption_metadata,
    )

    combined_dict: Dict = open_and_extract_coco_json(
        gcp_cracc = cracc,
        coco_blob = coco_blob
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
        image_folder_path: str = env_var.coco_train_image_folder

        rc_image_np: np.ndarray = preprocess_coco_image(
            gcp_cracc = cracc,
            bucket_name = coco_blob.bucket_name,
            image_folder_path = image_folder_path,
            image_file_name = image_file_name
        )

        print(f"resized_cropped_image {image_file_name} size {rc_image_np.shape}")

if __name__ == "__main__":
    main()
