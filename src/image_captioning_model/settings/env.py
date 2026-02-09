import os

from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()


@dataclass
class EnvVar:

    """
    Container object for project environment variables.

    This dataclass centralizes all environment variables required by the
    image-captioning pipeline so they can be accessed in a structured and
    type-safe way throughout the application.

    Attributes:
        gcs_coco_dataset (str):
            Path or URI to the COCO dataset stored in Google Cloud Storage.

        gcs_output_dir (str):
            Output directory in GCS where processed artifacts, models,
            or intermediate results will be stored.

        gcp_project_id (str):
            Google Cloud project identifier used for authentication
            and resource access.

        gcp_project_location (str):
            Default GCP region used for services such as Vertex AI
            or storage operations.

        coco_bucket_name (str):
            Name of the GCS bucket containing COCO dataset assets.

        coco_train_image_folder (str):
            Folder path inside the bucket containing training images.

        coco_train_label_metadata (str):
            Path to COCO label metadata JSON file.

        coco_train_caption_metadata (str):
            Path to COCO caption metadata JSON file.
    """

    gcs_coco_dataset: str
    gcs_output_bucket_name: str
    gcs_output_dir: str
    gcp_project_id: str
    gcp_project_location: str
    coco_bucket_name: str
    coco_train_image_folder: str
    coco_train_label_metadata: str
    coco_train_caption_metadata: str


def get_env(

) -> EnvVar:

    """
    Load environment variables required by the project.

    This function:
    1. Configures TensorFlow logging verbosity to reduce console noise.
    2. Reads required environment variables using `os.getenv`.
    3. Returns them as a structured `EnvVar` dataclass.

    If an environment variable is not defined, a fallback string
    indicating the missing variable is returned instead.

    Returns:
        EnvVar:
            A populated environment configuration object containing
            all required project environment variables.

    Notes:
        Expected environment variables:
        - GCS_COCO_DATASET
        - GCS_OUTPUT_DIR
        - GCP_PROJECT_ID
        - GCP_PROJECT_LOCATION
        - COCO_BUCKET_NAME
        - GCS_OUTPUT_BUCKET_NAME
        - COCO_TRAIN_IMAGE_FOLDER
        - COCO_TRAIN_LABEL_METADATA
        - COCO_TRAIN_CAPTION_METADATA
    """

    # Suppress warnings from TensorFlow
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # 0=all, 1=INFO off, 2=INFO+WARN off, 3=INFO+WARN+ERROR off (best effort)
    os.environ["AUTOGRAPH_VERBOSITY"] = "0"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # optional: reduces some CPU info logs


    # Read Environment Variables Pertaining to the Project
    GCS_COCO_DATASET: str = os.getenv("GCS_COCO_DATASET", "GCS_COCO_DATASET not found")
    GCS_OUTPUT_DIR: str = os.getenv("GCS_OUTPUT_DIR", "GCS_OUTPUT_DIR not found")
    GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "GCP_PROJECT_ID not found")
    GCP_PROJECT_LOCATION: str = os.getenv("GCP_PROJECT_LOCATION", "GCP_PROJECT_LOCATION not found")
    GCS_OUTPUT_BUCKET_NAME: str = os.getenv("GCS_OUTPUT_BUCKET_NAME", "GCS_OUTPUT_BUCKET_NAME not found")

    COCO_BUCKET_NAME: str = os.getenv("COCO_BUCKET_NAME", "COCO_BUCKET_NAME not found")
    COCO_TRAIN_IMAGE_FOLDER: str = os.getenv("COCO_TRAIN_IMAGE_FOLDER", "COCO_TRAIN_IMAGE_FOLDER not found")
    COCO_TRAIN_LABEL_METADATA: str = os.getenv("COCO_TRAIN_LABEL_METADATA", "COCO_TRAIN_LABEL_METADATA not found")
    COCO_TRAIN_CAPTION_METADATA:str = os.getenv("COCO_TRAIN_CAPTION_METADATA", "COCO_TRAIN_CAPTION_METADATA not found")

    return EnvVar(
        gcs_coco_dataset = GCS_COCO_DATASET,
        gcs_output_bucket_name = GCS_OUTPUT_BUCKET_NAME,
        gcs_output_dir = GCS_OUTPUT_DIR,
        gcp_project_id = GCP_PROJECT_ID,
        gcp_project_location = GCP_PROJECT_LOCATION,
        coco_bucket_name = COCO_BUCKET_NAME,
        coco_train_image_folder = COCO_TRAIN_IMAGE_FOLDER,
        coco_train_label_metadata = COCO_TRAIN_LABEL_METADATA,
        coco_train_caption_metadata = COCO_TRAIN_CAPTION_METADATA,
    )