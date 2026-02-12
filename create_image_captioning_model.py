from typing import List, Dict, Optional, Tuple, Union, Any

from pathlib import Path
import json
import tempfile
import gzip


from src.image_captioning_model.GCP.CredentialAccessor.CredentialAccessor import CredentialAccessor as CrAcc
from src.image_captioning_model.GCP.DaoCloudStorage.DaoCloudStorage import DaoCloudStorage
from src.image_captioning_model.settings.env import EnvVar, get_env
from src.image_captioning_model.settings.config import ProjectConfig, load_project_config

env_var: EnvVar = get_env()

project_root_path: Path = Path.cwd()
project_config: ProjectConfig = load_project_config(
    project_root_path = project_root_path,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Dense,
    Attention,
    Concatenate,
    GlobalAveragePooling2D,
    Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import (
    Tokenizer,
    text_to_word_sequence
)
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array
)
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences


import logging
tf.get_logger().setLevel(logging.ERROR)

from google.cloud.storage import Blob

# Defining Global Constants
EPOCHS: int = 20
BATCH_SIZE: int = 128
MAX_WORDS: int = 10_000
READ_IMAGES: int = 90_000
LAYER_SIZE: int = 256
EMBEDDING_WIDTH: int = 128
OOV_WORD: str = "UNK"
PAD_INDEX: int = 0
OOV_INDEX: int = 1
START_INDEX: int = MAX_WORDS - 2
STOP_INDEX: int = MAX_WORDS - 1
MAX_LENGTH: int = 60


def read_training_file(
    mygcs: DaoCloudStorage,
    bucket_name: str,
    caption_object_name: str,
    feature_vector_prefix: str
) -> Tuple[List[str], List[List[str]]]:

    def convert_jsonl_to_dict(
            local_json_path: Path
    ) -> Dict[int, List[str]]:

        result: Dict = {}

        with open(local_json_path, "r", encoding="utf-8") as f_handler:

            for each_line in f_handler:
                each_obj: Dict = json.loads(each_line)

                for the_key, the_val in each_obj.items():
                    result[int(the_key)] = the_val

        return result


    caption_dict: Dict = mygcs.process_blob_with_handler(
        bucket_name = bucket_name,
        object_name = caption_object_name,
        callback_func = convert_jsonl_to_dict
    )

    image_paths: List[str] = []
    dest_word_sequences: List[List[str]] = []

    for each_idx, key in enumerate(sorted(caption_dict.keys())):

        if each_idx >= READ_IMAGES:
            break

        image_thing: List[str] = caption_dict[key]

        if len(image_thing) < 2:
            # skip processing if the image has no caption
            continue

        image_path_str: str = image_thing[0]
        image_path_str = feature_vector_prefix + image_path_str
        image_paths.append(image_path_str)

        first_caption: str = image_thing[1]
        word_seq: List[str] = text_to_word_sequence(first_caption)
        a_dest_word_sequence = word_seq[0:MAX_LENGTH]
        dest_word_sequences.append(a_dest_word_sequence)

    return image_paths, dest_word_sequences


