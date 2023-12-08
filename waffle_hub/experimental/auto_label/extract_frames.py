import argparse
import os
from pathlib import Path
from typing import Union

import cv2
from tqdm import tqdm
from waffle_utils.file import io, search


def extract_frames(video_path: Union[str, Path], output_dir: Union[str, Path], interval: int = 1):
    video_path = Path(video_path)

    cap = cv2.VideoCapture(str(video_path))

    frame_dir = Path(output_dir, video_path.stem)
    io.make_directory(frame_dir)

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % interval == 0:
            frame_path = frame_dir / f"{frame_id:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
        frame_id += 1
    cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--interval", type=int, default=1)
    args = parser.parse_args()

    if Path(args.video_path).is_file():
        video_paths = [Path(args.video_path)]
    else:
        video_paths = search.get_video_files(args.video_path)

    for video_path in tqdm(video_paths):
        extract_frames(video_path, args.output_dir, args.interval)


if __name__ == "__main__":
    main()
