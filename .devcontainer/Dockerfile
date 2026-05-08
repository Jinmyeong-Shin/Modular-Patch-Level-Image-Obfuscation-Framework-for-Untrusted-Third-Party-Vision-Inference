FROM --platform=linux/amd64 docker.io/python:3.14-trixie

ENV DEBIAN_FRONTEND=noninteractive

# Base utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        curl \
        ca-certificates \
        gnupg \
        git \
        sudo \
    && rm -rf /var/lib/apt/lists/*

# CUDA 13.2 runtime
RUN wget -qO /tmp/cuda-keyring.deb \
        https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && rm /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends cuda-runtime-13-2 \
    && rm -rf /var/lib/apt/lists/*

# Python virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Pip baseline (installs into the venv)
RUN pip install --upgrade pip
