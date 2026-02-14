from typing import List, Dict, Optional, Tuple, Union, Any, Set

from pathlib import Path
import json
import tempfile
import gzip
from pprint import pprint

import numpy as np

from src.image_captioning_model.GCP.CredentialAccessor.CredentialAccessor import CredentialAccessor as CrAcc
from src.image_captioning_model.GCP.DaoCloudStorage.DaoCloudStorage import DaoCloudStorage
from src.image_captioning_model.vision.preprocess import resize_and_crop_image
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
from tensorflow.python.keras.callbacks import History
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
# BATCH_SIZE: int = 128
BATCH_SIZE: int = 10
MAX_WORDS: int = 10_000
# READ_IMAGES: int = 90_000
READ_IMAGES: int = 100
LAYER_SIZE: int = 256
EMBEDDING_WIDTH: int = 128

PAD_WORD: str = "PAD"
OOV_WORD: str = "UNK"
START_WORD: str = "START"
STOP_WORD: str = "STOP"

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

    """Read image paths and caption word sequences from a JSONL captions blob.

    The captions file is expected to be JSON Lines where each line is a JSON
    object that maps an image ID to a list of strings. The list's first element
    is treated as the image path, and the second element is treated as the
    caption text. Only the first caption is used, and captions are tokenized
    with `text_to_word_sequence`, then truncated to `MAX_LENGTH` tokens.

    Image paths are prefixed with `feature_vector_prefix`, and processing is
    limited to the first `READ_IMAGES` image IDs in sorted order. Entries with
    fewer than two elements (missing captions) are skipped.

    Args:
        mygcs: Cloud storage DAO used to read the captions blob.
        bucket_name: GCS bucket containing the captions JSONL object.
        caption_object_name: Object name for the captions JSONL file.
        feature_vector_prefix: Prefix to prepend to each image path.

    Returns:
        A tuple of (image_paths, dest_word_sequences).
        - image_paths: list of prefixed image path strings.
        - dest_word_sequences: list of token lists for the first caption.
    """

    def convert_jsonl_to_dict(
            local_json_path: Path
    ) -> Dict[int, List[str]]:

        result: Dict = {}

        with gzip.open(local_json_path, "rt", encoding="utf-8") as f_handler:

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


def tokenize(
        list_sentence_phrase: Union[List[str], List[List[str]]]
) -> Tuple:
    """
    Tokenizes a list of sentences or phrases into sequences of integer tokens.

    This function utilizes a tokenizer to convert a list of input sentences or
    phrases into corresponding numerical token sequences. The tokenizer is
    configured to handle a maximum number of words and assign a special token
    for out-of-vocabulary words.

    Args:
        list_sentence_phrase (Union[List[str], List[List[str]])): A list of strings, where each string is a
            sentence or phrase that needs to be tokenized.

    Returns:
        Tuple[Tokenizer, List[List[int]]]: A tuple containing the tokenizer instance
        and the tokenized sequences. The tokenizer instance is configured with
        predefined settings, and the token sequences are a list of lists where each
        inner list represents the tokenized form of a corresponding sentence or
        phrase from the input.
    """
    tokenizer: Tokenizer = Tokenizer(
        num_words = MAX_WORDS - 2,
        oov_token = OOV_WORD
    )

    tokenizer.fit_on_texts(list_sentence_phrase)
    token_sequences: List[List[int]] = tokenizer.texts_to_sequences(list_sentence_phrase)

    return tokenizer, token_sequences


def token_to_words(
        tokenizer: Tokenizer,
        token_sequence: List[int]
) -> List[str]:
    """
    Converts a sequence of token indices to their corresponding word representations
    based on the specified tokenizer.

    This function takes a list of token indices and maps them to their respective
    words using the given tokenizer. Special indices like PAD_INDEX, OOV_INDEX,
    START_INDEX, and STOP_INDEX are mapped to their predefined word representations.

    Args:
        tokenizer (Tokenizer): The tokenizer object used to convert token indices
            to corresponding words.
        token_sequence (List[int]): A list of integers representing the token
            indices to be converted into words.
    """

    word_sequence: List[str] = []

    for each_index in token_sequence:

        if each_index == PAD_INDEX:
            word_sequence.append(PAD_WORD)
        elif each_index == OOV_INDEX:
            word_sequence.append(OOV_WORD)
        elif each_index == START_INDEX:
            word_sequence.append(START_WORD)
        elif each_index == STOP_INDEX:
            word_sequence.append(STOP_WORD)
        else:
            word_sequence.append(
                tokenizer.sequences_to_texts([[int(each_index)]])[0]
            )

    return word_sequence


def token_to_sentence(
        tokenizer: Tokenizer,
        token_sequence: List[int]
) -> str:
    """
    Converts a sequence of tokens into a sentence string, filtering out specific tokens
    and handling duplicated consecutive words.

    Args:
        tokenizer (Tokenizer): The tokenizer used to map token IDs to words.
        token_sequence (List[int]): A list of integer token IDs representing a
            sequence of words.

    Returns:
        str: A string representing the processed sentence.
    """
    skip_words: Set[str] = {PAD_WORD, OOV_WORD, START_WORD, STOP_WORD}
    prev_word: Optional[str] = None

    list_words: List[str] = token_to_words(tokenizer, token_sequence)

    result_words: List[str] = []

    for each_word in list_words:

        if each_word == STOP_WORD:
            break
        elif each_word not in skip_words:
            continue
        elif each_word == prev_word:
            continue

        result_words.append(each_word)
        prev_word = each_word

    sentence: str = " ".join(result_words)

    return sentence


class ImageCaptionSequence(Sequence):
    """
    Handles batching and preprocessing of image-caption data for use in sequence models.

    This class is designed to manage input and output data for training or inference
    in sequence-based models. It processes batches of image paths, input captions,
    and target captions, supporting integration with storage solutions and efficient
    data retrieval.

    Attributes:
        image_paths (List[str]): List of file paths corresponding to image data.
        dest_input_data (List[List[str]]): List of tokenized input caption sequences.
        dest_target_data (List[List[str]]): List of tokenized target caption sequences.
        batch_size (int): Number of samples per batch during data loading.
        mygcs (DaoCloudStorage): Instance of a data access object for Google Cloud
            Storage used to fetch and handle data blobs.
        bucket_name (str): Name of the storage bucket from which resources are retrieved.
    """
    def __init__(
            self,
            image_paths: List[str],
            dest_input_data: List[List[str]],
            dest_target_data: List[List[str]],
            batch_size: int,
            mygcs: DaoCloudStorage,
            bucket_name: str,
    ) -> None:

        self.image_paths: List[str] = image_paths
        self.dest_input_data: List[List[str]] = dest_input_data
        self.dest_target_data: List[List[str]] = dest_target_data
        self.batch_size: int = batch_size
        self.mygcs: DaoCloudStorage = mygcs
        self.bucket_name: str = bucket_name

    def __len__(
            self
    ) -> int:
        """
        Calculates the total number of batches based on the input data and batch size.

        This method computes the number of batches required to process the input data,
        taking into account the batch size and rounding up to the nearest whole number
        to accommodate any remainder.

        Returns:
            int: The total number of batches needed to process the input data.
        """
        return int(
            np.ceil(len(self.dest_input_data)) / float(self.batch_size)
        )


    def _load_npz(
        self,
        object_name: str
    ) -> np.ndarray:
        """
        Loads an NPZ file from Google Cloud Storage (GCS) and extracts the
        'block5_conv4' dataset.

        This method appends the ".npz" file extension to `object_name` if
        not already present, fetches the NPZ from a GCS bucket, and retrieves
        a specific dataset (key: "block5_conv4") within the NPZ file.

        Args:
            object_name (str): The name of the NPZ file object stored in GCS.
                If the name does not end with ".npz", the extension will be
                appended automatically.

        Returns:
            np.ndarray: The extracted dataset corresponding to the key
            'block5_conv4' from the NPZ file.
        """

        if not object_name.endswith(".npz"):
            object_name = object_name + ".npz"

        def _read(
            local_path: Path
        ) -> np.ndarray:
            with np.load(local_path) as npz_handler:
                arr = npz_handler["block5_conv4"]

            return arr.squeeze(0)

        return self.mygcs.process_blob_with_handler(
            bucket_name = self.bucket_name,
            object_name = object_name,
            callback_func = _read
        )

    def __getitem__(
            self,
            idx: int
    ) -> Tuple[List[Union[np.ndarray, List[str]]], List[List[str]]]:
        """
        Retrieves a batch of data for a specified index.

        This method is used to fetch a batch of features and target data for the
        given batch index. The batch is constructed by slicing the respective
        data arrays using the current batch index and batch size.

        Args:
            idx (int): The index of the batch to retrieve.

        Returns:
            Tuple[List[Union[np.ndarray, List[str]]], List[List[str]]]: A tuple where
            the first element is a list containing the batch of features as a NumPy
            array and the associated input data as a list of lists. The second
            element is a list of target data corresponding to the batch.
        """
        batch_x0: List[str] = self.image_paths[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        batch_x1: List[List[str]] = self.dest_input_data[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        batch_y: List[List[str]] = self.dest_target_data[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]

        features: np.ndarray = np.stack(
            [self._load_npz(each_img) for each_img in batch_x0], axis=0
        )

        return [features, batch_x1], batch_y

def build_encoder_model(

) -> Model:
    """
    Builds and returns an encoder model using TensorFlow/Keras layers.

    This function creates an encoder model that processes a feature vector input by
    applying global average pooling, followed by fully connected dense layers to
    produce the encoding states. The output contains two dense layer outputs
    representing encoded states, which can be used as inputs for subsequent
    decoder layers or other downstream tasks.

    Returns:
        keras.Model: The constructed encoder model.
    """
    # Input is the feature vector
    feature_vector_input: tf.Tensor = Input(shape=(14,14, 512))

    # Create encoder layers
    enc_mean_layer: tf.Module = GlobalAveragePooling2D()
    enc_layer_h: tf.Module = Dense(LAYER_SIZE)
    enc_layer_c: tf.Module = Dense(LAYER_SIZE)

    # Connect the encoding layer
    enc_mean_layer_output: tf.Tensor = enc_mean_layer(feature_vector_input)
    enc_layer_h_outputs: tf.Tensor = enc_layer_h(enc_mean_layer_output)
    enc_layer_c_outputs: tf.Tensor = enc_layer_c(enc_mean_layer_output)

    # Organize the output state for encoder layers
    enc_layer_outputs: List[tf.Tensor] = [enc_layer_h_outputs, enc_layer_c_outputs]

    # Build the model
    enc_model_top: Model = Model(
        inputs = feature_vector_input,
        outputs = enc_layer_outputs
    )

    enc_model_top.summary()

    return enc_model_top



def build_inference_encoder_model(
    encoder_model: Model
) -> Model:
    """
    Builds an inference encoder model by combining a pre-trained convolutional model
    and a specified encoder model, facilitating transfer learning for tasks such as
    feature extraction and intermediate representation generation.

    Args:
        encoder_model: The encoder model to be used for processing the output of
            the convolutional model. It must be callable and return an intermediate
            state when applied to the outputs of the convolutional model.

    Returns:
        Model: A Keras Model that takes an input tensor, passes it through the
        convolutional model, processes its output using the encoder model, and
        returns both the intermediate state and the convolutional model outputs.
    """

    conv_model: tf.Module = VGG19(weights = "imagenet")
    conv_model_outputs: tf.Tensor = conv_model.get_layer("block5_conv4").output

    intermediate_state: tf.Tensor = encoder_model(conv_model_outputs)

    inference_enc_model: Model = Model(
        [conv_model.input],
        intermediate_state + [conv_model_outputs]
    )

    inference_enc_model.summary()

    return inference_enc_model


def build_decoder_model(

) -> Model:
    """
    Builds and returns a decoder model for sequence generation with an attention mechanism.

    The decoder model is designed to work with a feature vector representation of an image and
    a sequence of input tokens (image captions). It uses an LSTM-based architecture with an
    attention mechanism to generate word predictions. The model outputs the predicted
    token probabilities and intermediate LSTM states.

    Returns:
        Model: A TensorFlow/Keras model of the decoder with specified input and output configurations.

    Args:
        None

    Raises:
        None

    Inputs:
        - feature_vector_input (tf.Tensor): Tensor with shape (14,14,512) representing the
          spatial feature map of the image.
        - embedding_input (tf.Tensor): Tensor with shape (None,) representing the tokenized
          input sequence (image caption sequence).
        - layer1_state_input_h (tf.Tensor): Tensor with shape (LAYER_SIZE,) representing the
          hidden state of the LSTM's first layer.
        - layer1_state_input_c (tf.Tensor): Tensor with shape (LAYER_SIZE,) representing the
          cell state of the LSTM's first layer.

    Outputs:
        - output_probabilities (tf.Tensor): Tensor with shape (MAX_WORDS,) representing the
          softmax probabilities for predicted tokens.
        - state_h (tf.Tensor): Tensor with shape (LAYER_SIZE,) representing the hidden state
          of the LSTM's first layer after processing the input.
        - state_c (tf.Tensor): Tensor with shape (LAYER_SIZE,) representing the cell state
          of the LSTM's first layer after processing the input.

    Note:
        - The model incorporates an attention mechanism to selectively focus on different parts
          of the feature vector.
        - The embedding layer maps input token indices to dense vectors.
        - The LSTM layer processes the sequences and maintains intermediate states for continuity.
    """

    # Input to the network is:
    # - the feature_vector
    # - image_caption_sequence
    # - intermediate_state
    dec_feature_vector_input: tf.Tensor = Input(shape=(14,14, 512))
    dec_embedding_input: tf.Tensor = Input(shape=(None,))
    dec_layer1_state_input_h: tf.Tensor = Input(shape=(LAYER_SIZE,))
    dec_layer1_state_input_c: tf.Tensor = Input(shape=(LAYER_SIZE,))

    # Create the decoder layers
    dec_reshape_layer: tf.Module = Reshape(
        (196, 512),
        input_shape = (14, 14, 512, )
    )
    dec_attention_layer: tf.Module = Attention()
    dec_query_layer: tf.Module = Dense(512)
    dec_embedding_layer: tf.Module = Embedding(
        output_dim = EMBEDDING_WIDTH,
        input_dim = MAX_WORDS,
        mask_zero = False
    )

    dec_layer1: tf.Module = LSTM(
        LAYER_SIZE,
        return_state = True,
        return_sequences = True
    )

    dec_concat_layer: tf.Module = Concatenate()

    dec_layer2: tf.Module = Dense(MAX_WORDS, activation="softmax")

    # Connect the decoder layers
    dec_embedding_layer_outputs: tf.Tensor = dec_embedding_layer(dec_embedding_input)
    dec_reshape_layer_outputs: tf.Tensor = dec_reshape_layer(dec_feature_vector_input)
    dec_layer1_outputs, dec_layer1_state_h, dec_layer1_state_c =  dec_layer1(
        dec_embedding_layer_outputs,
        initial_state = [
            dec_layer1_state_input_h,
            dec_layer1_state_input_c
        ]
    )
    dec_query_layer_outputs: tf.Tensor = dec_query_layer(dec_layer1_outputs)
    dec_attention_layer_outputs: tf.Tensor = dec_attention_layer(
        [
            dec_query_layer_outputs,
            dec_reshape_layer_outputs
        ]
    )
    dec_layer2_inputs: tf.Tensor = dec_concat_layer(
        [
            dec_layer1_outputs,
            dec_attention_layer_outputs
        ]
    )

    dec_layer2_outputs: tf.Tensor = dec_layer2(dec_layer2_inputs)

    # Build the model
    dec_model: Model = Model(
        inputs = [
            dec_feature_vector_input,
            dec_embedding_input,
            dec_layer1_state_input_h,
            dec_layer1_state_input_c
        ],
        outputs = [
            dec_layer2_outputs,
            dec_layer1_state_h,
            dec_layer1_state_c
        ]
    )

    dec_model.summary()
    return dec_model


def build_training_full_encoder_decoder_model(
    encoder_model: Model,
    decoder_model: Model
) -> Model:
    """
    Builds and compiles a full training model combining an encoder and a decoder.

    This function takes an encoder model and a decoder model as inputs and constructs a
    full training model. The full model integrates the encoder and decoder to pass data
    through sequentially. It compiles the model using the Adam optimizer and sparse
    categorical crossentropy loss for multi-class classification tasks. The function
    summarizes the training model's architecture before returning it.

    Args:
        encoder_model (Model): The encoder model instance to process the input feature
            vector.
        decoder_model (Model): The decoder model instance to process the decoder
            embeddings and generate predictions.

    Returns:
        Model: The compiled full training model integrating the encoder and decoder.
    """
    # Build and compile full training model
    train_feature_vector_input: tf.Tensor = Input(shape=(14,14, 512))
    train_dec_embedding_input: tf.Tensor = Input(shape=(None,))

    intermediate_state: tf.Tensor = encoder_model(train_feature_vector_input)
    train_decoder_output, _, _ = decoder_model(
        [
            train_feature_vector_input,
            train_dec_embedding_input,
        ] + intermediate_state
    )

    training_model: Model = Model(
        inputs = [
            train_feature_vector_input,
            train_dec_embedding_input,
        ],
        outputs = [train_decoder_output]
    )

    training_model.compile(
        loss = "sparse_categorical_crossentropy",
        optimizer = "adam",
        metrics = ["accuracy"]
    )

    training_model.summary()

    return training_model


def fetch_test_images(
        mygcs: DaoCloudStorage,
        bucket_name: str,
) -> List[Path]:
    """
    Fetches and returns paths to test images from a specified bucket in cloud storage.

    This function downloads files with the ".jpg" extension from a specific prefix
    within a cloud storage bucket to a temporary directory. It then retrieves the
    paths of the downloaded test images and returns them in sorted order. If the
    downloaded directory is empty or None, it returns an empty list.

    Args:
        mygcs: An instance of DaoCloudStorage, representing the interface to interact
            with the cloud storage service.
        bucket_name: The name of the cloud storage bucket from which test images are
            to be downloaded.

    Returns:
        List[Path]: A sorted list of file paths pointing to the downloaded test image
            files. If no files are found, an empty list is returned.
    """
    tmp_dir: Path = mygcs.download_prefix_to_temp(
        bucket_name = bucket_name,
        prefix = "test_images",
        allowed_extensions = [".jpg"]
    )

    if tmp_dir is None:
        return []

    return sorted(tmp_dir.glob("*.jpg"))


def main(

) -> None:

    cwd: Path = Path.cwd()
    credential_path: Path = cwd / "credentials" / "gcp" / "service_account.json"

    cracc: CrAcc = CrAcc(
        credential_path = credential_path,
        project_id = env_var.gcp_project_id,
        location = env_var.gcp_project_location,
    )

    mygcs: DaoCloudStorage = DaoCloudStorage(
        credential_accessor = cracc
    )

    test_image_paths: List[Path] = fetch_test_images(
        mygcs = mygcs,
        bucket_name = env_var.gcs_output_bucket_name
    )

    feature_vector_prefix: str = f"{env_var.gcs_output_dir.rstrip('/')}/vgg19_block5_conv4/"

    image_paths, dest_word_sequences = read_training_file(
        mygcs = mygcs,
        bucket_name = env_var.gcs_output_bucket_name,
        caption_object_name = env_var.gcs_output_caption_object_name,
        feature_vector_prefix = feature_vector_prefix
    )

    dest_tokenizer, dest_token_sequences = tokenize(dest_word_sequences)

    # prepare training data
    dest_target_token_sequence: List[List[int]] = [
        each_token + [STOP_INDEX]
        for each_token in dest_token_sequences
    ]

    dest_input_token_sequence: List[List[int]] = [
        [START_INDEX] + each_token
        for each_token in dest_target_token_sequence
    ]

    dest_input_data = pad_sequences(
        dest_input_token_sequence,
        padding = "post",
    )

    dest_target_data = pad_sequences(
        dest_target_token_sequence,
        padding = "post",
        maxlen = len(dest_input_data[0])
    )

    image_caption_sequence = ImageCaptionSequence(
        image_paths = image_paths,
        dest_input_data = dest_input_data,
        dest_target_data = dest_target_data,
        batch_size = BATCH_SIZE,
        mygcs = mygcs,
        bucket_name = env_var.gcs_output_bucket_name
    )

    enc_model_top: Model = build_encoder_model()
    decoder_model: Model = build_decoder_model()

    training_model: Model = build_training_full_encoder_decoder_model(
        encoder_model = enc_model_top,
        decoder_model = decoder_model
    )

    inference_enc_model: Model = build_inference_encoder_model(
        encoder_model=enc_model_top
    )

    break_limit: int = 5

    for each_epoch in range(EPOCHS):

        if each_epoch >= break_limit:
            break

        print("#"*60)
        print(f"Epoch {each_epoch}/{EPOCHS}")
        print("#"*60)

        history: History = training_model.fit(
            image_caption_sequence,
            epochs = 1,
            verbose = 2
        )

        # Try to make caption from test images
        for each_img_path in test_image_paths:

            image_file_name: str = each_img_path.name

            rc_image_np: np.ndarray = mygcs.stream_blob_with_handler(
                bucket_name = env_var.gcs_output_bucket_name,
                object_name = (
                    env_var.gcs_output_test_image_folder +
                    image_file_name
                ),
                callback_func = resize_and_crop_image
            )

            x = preprocess_input(rc_image_np)
            dec_layer1_state_h, dec_layer1_state_c, feature_vector = (
                inference_enc_model.predict(x, verbose = 2)
            )

            # predict sentence word-for-word
            prev_word_index: int = START_INDEX
            generated_sentence: List[str] = []
            pred_seq: List[int] = []

            for each_jdx in range(MAX_LENGTH):
                x = np.reshape(
                    np.array(prev_word_index),
                    (1, 1)
                )

                preds, dec_layer1_state_h, dec_layer1_state_c = (
                    decoder_model.predict(
                        [
                            feature_vector,
                            x,
                            dec_layer1_state_h,
                            dec_layer1_state_c
                        ],
                        verbose = 0
                    )
                )

                prev_word_index = np.asarray(
                    preds[0][0]
                ).argmax()
                pred_seq.append(prev_word_index)

                if prev_word_index == STOP_INDEX:
                    break

            the_caption: str = token_to_sentence(
                tokenizer = dest_tokenizer,
                token_sequence = pred_seq,
            )

            image_caption_dict: Dict[str, str] = {
                "test_image_name": image_file_name,
                "generated_caption": the_caption
            }

            pprint(
                image_caption_dict,
                indent = 4
            )




if __name__ == "__main__":
    main()