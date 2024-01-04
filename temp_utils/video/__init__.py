# it will be deleted after waffle_dough is released

import cv2

SUPPORTED_VIDEO_EXTENSION = ["mp4", "avi", "wmv", "mov", "flv", "mkv", "mpeg"]
DEFAULT_VIDEO_EXTENSION = SUPPORTED_VIDEO_EXTENSION[0]
FOURCC_MAP = {
    "mp4": cv2.VideoWriter_fourcc(*"mp4v"),
    "avi": cv2.VideoWriter_fourcc(*"xvid")
    if cv2.VideoWriter_fourcc(*"xvid") == -1
    else cv2.VideoWriter_fourcc(*"mjpg"),
    "wmv": cv2.VideoWriter_fourcc(*"wmv2"),
    "mov": cv2.VideoWriter_fourcc(*"xvid"),
    "flv": cv2.VideoWriter_fourcc(*"flv1"),
    "mkv": cv2.VideoWriter_fourcc(*"vp80"),
    "mpeg": cv2.VideoWriter_fourcc(*"xvid"),
    "mpg": cv2.VideoWriter_fourcc(*"xvid"),
}


def get_fourcc(extension: str) -> cv2.VideoWriter_fourcc:
    """Get OpenCV fourcc by extension name.
    Args:
        extension (str): extension name. without '.'.
    Raises:
        KeyError: if extension is not supported.
    Returns:
        cv2.VideoWriter_fourcc: OpenCV fourcc
    """
    if extension not in FOURCC_MAP:
        raise KeyError(f"{extension} is not supported. Choose one of {list(FOURCC_MAP.keys())}")
    return FOURCC_MAP[extension]
