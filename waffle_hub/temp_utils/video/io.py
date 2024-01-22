from pathlib import Path
from typing import Tuple, Union

import cv2
import imageio

from waffle_hub.temp_utils.video import get_fourcc


def create_video_capture(input_path: Union[str, Path]) -> Tuple[cv2.VideoCapture, dict]:
    """
    Create a VideoCapture object and retrieve video metadata.

    This function opens a video file using OpenCV's VideoCapture class and retrieves
    the video's frames per second, width, and height.

    Args:
        input_path (Union[str, Path]): The path to the input video file.

    Returns:
        Tuple[cv2.VideoCapture, dict]: A tuple containing the VideoCapture object
                                       and a dictionary with the video metadata.

    Raises:
        ValueError: If the video file cannot be opened.
    """

    # Create a VideoCapture object using the input path
    cap = cv2.VideoCapture(str(input_path))

    # Check if the video file was opened successfully
    if not cap.isOpened():
        raise ValueError(f"Failed to open the video file at {input_path}. Check the input path.")

    # Retrieve the video metadata (fps, width, height)
    ext = Path(input_path).suffix[1:]
    if ext == "mkv":
        # Use imageio to retrieve the video metadata for mkv files
        # (OpenCV has a bug that causes it to return incorrect metadata for mkv files)
        video_reader = imageio.get_reader(str(input_path))
        video_meta = video_reader.get_meta_data()
        fps = video_meta["fps"]
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)

    meta = {
        "fps": fps,
        "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
    }

    # Return the VideoCapture object and the metadata
    return cap, meta


def create_video_writer(
    output_path: Union[str, Path],
    fps: float,
    frame_size: tuple[int, int],
) -> cv2.VideoWriter:

    fourcc = get_fourcc(Path(output_path).suffix[1:])
    return cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
