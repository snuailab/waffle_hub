from pathlib import Path

import pytest
from waffle_utils.file import io, network


@pytest.fixture(scope="session")
def coco_path(tmp_path_factory: pytest.TempPathFactory):
    url = "https://raw.githubusercontent.com/snuailab/assets/main/waffle/sample_dataset/mnist.zip"

    tmpdir = tmp_path_factory.mktemp("coco")
    zip_file = tmpdir / "mnist.zip"
    coco_path = tmpdir / "extract"

    network.get_file_from_url(url, zip_file, create_directory=True)
    io.unzip(zip_file, coco_path, create_directory=True)

    return Path(coco_path)
