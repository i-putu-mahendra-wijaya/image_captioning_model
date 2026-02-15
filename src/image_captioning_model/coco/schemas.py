from dataclasses import dataclass

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