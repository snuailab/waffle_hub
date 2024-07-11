import logging
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union

import datumaro as dm
from waffle_utils.file import io, search

from .coco import export_coco


def export_datumaro(self, export_dir: Union[str, Path]) -> str:
    """Export dataset to Datumaro format

    Args:
        export_dir (Union[str, Path]): Path to export directory

    Returns:
        str: Path to export directory
    """
    export_dir = Path(export_dir)

    coco_export_dir = Path(export_dir).with_name("COCO")
    logging.info("Export dataset to COCO format (for Datumaro)")
    export_coco(self, coco_export_dir)

    with TemporaryDirectory() as coco_temp:
        io.copy_files_to_directory(coco_export_dir, Path(coco_temp))
        io.move_files_to_directory(
            src=search.get_files(Path(coco_temp), extension=".json"),
            dst=Path(coco_temp) / "annotations",
            create_directory=True,
            recursive=False,
        )

        dm_dataset = dm.Dataset.import_from(coco_temp, "coco")
        dm_dataset.export(export_dir, "datumaro", save_media=True)

    return export_dir
