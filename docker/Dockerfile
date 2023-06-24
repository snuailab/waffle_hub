FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt update \
    && apt install --no-install-recommends -y gcc git zip curl htop libgl1-mesa-glx libglib2.0-0 libpython3-dev gnupg g++

ENV OMP_NUM_THREADS=1

# Install Python packages (no cache)
ARG NO_CACHE_ARG

RUN pip install -U pip \
    && pip install -U git+https://github.com/snuailab/waffle_hub.git \
    && pip install -U numpy


WORKDIR /workspace