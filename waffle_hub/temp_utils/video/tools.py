import logging
from pathlib import Path
from typing import Optional, Union

from temp_utils.image import (
    DEFAULT_IMAGE_EXTENSION,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from temp_utils.image.io import load_image, save_image
from temp_utils.video.io import create_video_capture, create_video_writer
from waffle_utils.file.io import make_directory
from waffle_utils.file.search import get_image_files

logger = logging.getLogger(__name__)

DEFAULT_FRAME_RATE = 30


def extract_frames(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    num_of_frames: Optional[int] = None,
    interval_second: Optional[float] = None,
    output_image_extension: str = DEFAULT_IMAGE_EXTENSION,
    verbose: bool = False,
) -> None:
    f"""
    Extract frames from a video file at specified time intervals and save them as images.

    Args:
        input_path: The path to the input video file.
        output_dir: The directory where the extracted frames will be saved.
        num_of_frames: The number of frames to extract (optional). If not specified, all frames at the specified intervals will be extracted.
        interval_second: The time interval in seconds between extracted frames (optional). If not specified, all frames will be extracted.
        output_image_extension: The file extension for the output images (default: {DEFAULT_IMAGE_EXTENSION}).
        verbose: If True, print progress information (default: False).

    Raises:
        ValueError: If the output image extension is not supported.
    """

    # Convert input_path and output_dir to Path objects
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    # Check if the output image extension is supported
    if output_image_extension not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Invalid output_image_extension: {output_image_extension}.\n"
            f"Must be one of {SUPPORTED_IMAGE_EXTENSIONS}."
        )

    # Create output directory if it doesn't exist
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Extract frames from the video file
    video_capture, meta = create_video_capture(input_path)

    # Carculate frame interval
    frame_interval = int(meta["fps"] * interval_second) if interval_second else 1

    # Save frames as images
    count = 0
    extracted_frames = 0
    while True:
        success, image = video_capture.read()
        if not success:
            break

        # Only extract frames at the specified frame rate
        if count % frame_interval == 0:
            if verbose:
                logger.info(f"{input_path} -> {output_dir}/frame_{count}.{output_image_extension}")
            save_image(output_dir / f"frame_{count}.{output_image_extension}", image)
            extracted_frames += 1

            # Stop extracting frames if the specified number of frames is reached
            if num_of_frames is not None and extracted_frames >= num_of_frames:
                break

        count += 1

    # Release the video capture
    video_capture.release()
    logger.info(f"Output: {output_dir}/")


def create_video(
    input_dir: Union[str, Path],
    output_path: Union[str, Path],
    frame_rate: int = DEFAULT_FRAME_RATE,
    verbose: bool = False,
) -> None:
    f"""
    Creates a video file from a directory of frame images.

    Args:
        input_dir (Union[str, Path]): Path to the input directory containing the frame images.
        output_path (Union[str, Path]): Path to the output video file.
        frame_rate (int, optional): Frame rate of the output video. Defaults to {DEFAULT_FRAME_RATE}.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    """
    # Convert input_dir and output_path to Path objects
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    # Get image files
    image_files = get_image_files(input_dir)

    # Create output directory if it doesn't exist
    if not output_path.parent.exists():
        make_directory(input_dir)

    # Load the first frame to get dimensions
    first_frame = load_image(image_files[0])
    height, width = first_frame.shape[:2]

    # Determine the appropriate fourcc codec for the output video format
    out = create_video_writer(output_path, frame_rate, (width, height))

    # Iterate through frames and write to the video file
    for i, frame in enumerate(image_files):
        if verbose:
            logger.info(f"{frame} -> {output_path} ({i+1}/{len(image_files)})")
        image = load_image(frame)
        out.write(image)

    # Release the video writer
    out.release()
    logger.info(f"Output: {output_path}")
