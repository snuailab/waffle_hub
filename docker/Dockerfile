FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt update \
    && apt install --no-install-recommends -y gcc git zip curl htop libgl1-mesa-glx libglib2.0-0 libpython3-dev gnupg g++ wget

ENV OMP_NUM_THREADS=1

# Install somethings
RUN wget https://ultralytics.com/assets/Arial.Unicode.ttf https://ultralytics.com/assets/Arial.ttf && \
    mkdir -p /root/.config/Ultralytics && \
    cp Arial.Unicode.ttf /root/.config/Ultralytics && \
    cp Arial.ttf /root/.config/Ultralytics && \
    rm Arial.Unicode.ttf Arial.ttf

ENV WAFFLE_HUB_ROOT_DIR=/workspace/hubs
ENV WAFFLE_DATASET_ROOT_DIR=/workspace/datasets

# Install Python packages (no cache)
ARG NO_CACHE_ARG

WORKDIR /workspace
COPY . .
RUN pip install -U pip \
    && pip install -U . \
    && pip install -U numpy pytest
