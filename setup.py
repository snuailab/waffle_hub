# Waffle Hub (based on Ultralytics YOLO 🚀, GPL-3.0 license)

import re
from pathlib import Path

import pkg_resources as pkg
from setuptools import find_packages, setup

# Settings
FILE = Path(__file__).resolve()
PARENT = FILE.parent  # root directory
README = (PARENT / "README.md").read_text(encoding="utf-8")
REQUIREMENTS = [
    f"{x.name}{x.specifier}"
    for x in pkg.parse_requirements((PARENT / "requirements.txt").read_text())
]
PKG_REQUIREMENTS = ["sentry_sdk"]  # pip-only requirements


def get_version():
    file = PARENT / "waffle_hub/__init__.py"
    return re.search(
        r'^__version__ = [\'"]([^\'"]*)[\'"]',
        file.read_text(encoding="utf-8"),
        re.M,
    )[1]


setup(
    name="waffle_hub",  # name of pypi package
    version=get_version(),  # version of pypi package
    python_requires=">=3.9",
    license="GPL-3.0",
    description="Waffle hub",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/snuailab/waffle_hub",
    project_urls={
        "Bug Reports": "https://github.com/snuailab/waffle_hub/issues",
        "Source": "https://github.com/snuailab/waffle_hub",
    },
    author="SNUAILAB",
    author_email="huijae.lee@snuailab.ai",
    packages=find_packages(),  # required
    include_package_data=True,
    install_requires=REQUIREMENTS + PKG_REQUIREMENTS,
    # extras_require={
    #     'dev': ['check-manifest', 'pytest', 'pytest-cov', 'coverage', 'mkdocs-material', 'mkdocstrings[python]'],
    #     'export': ['coremltools>=6.0', 'onnx', 'onnxsim', 'onnxruntime', 'openvino-dev>=2022.3'],
    #     'tf': ['onnx2tf', 'sng4onnx', 'tflite_support', 'tensorflow']},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        # 'Intended Audience :: Education',
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        # 'Programming Language :: Python :: 3.11',
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Operating System :: POSIX :: Linux",
        # 'Operating System :: MacOS',
        # 'Operating System :: Microsoft :: Windows',
    ],
    keywords="machine-learning, deep-learning, vision, ML, DL, AI, YOLO, Ultralytics, SNUAILAB",
    entry_points={
        "console_scripts": [
            "waffle_utils = waffle_utils.run:app",
            "wu = waffle_utils.run:app",
            "waffle_dataset = waffle_hub.dataset.cli:main",
            "wd = waffle_hub.dataset.cli:main",
            "waffle_hub = waffle_hub.hub.cli:main",
            "wh = waffle_hub.hub.cli:main",
        ]
    },
)
