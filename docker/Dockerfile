FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt update \
    && apt install --no-install-recommends -y gcc git zip curl htop libgl1-mesa-glx libglib2.0-0 libpython3-dev gnupg g++

RUN pip install -U pip \
    && pip install -U git+https://github.com/snuailab/waffle_hub.git

WORKDIR /workspace