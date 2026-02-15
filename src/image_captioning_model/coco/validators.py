from typing import Dict, Tuple, List


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
