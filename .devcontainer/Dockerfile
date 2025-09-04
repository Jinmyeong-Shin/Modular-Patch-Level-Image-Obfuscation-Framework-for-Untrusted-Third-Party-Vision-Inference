FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV DPKG_MAINTSCRIPT_PACKAGE_WISE_STRICT_NOINITRAMFS=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    lsb-release \
    gnupg \
    software-properties-common \
    wget \
    git \
    python3 \
    python3-dev \
    python3-venv

# Install CUDA if specified
ARG CUDA_KEYRING_VERSION
RUN if [ -n "$CUDA_KEYRING_VERSION" ]; then \
    wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_${CUDA_KEYRING_VERSION}_all.deb && \
    dpkg -i cuda-keyring_${CUDA_KEYRING_VERSION}_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    cuda-toolkit && \
    apt-get clean -y; \
    rm -rf cuda-keyring_${CUDA_KEYRING_VERSION}_all.deb; \
    fi

RUN rm -rf /var/lib/apt/lists/*

# add CUDA to PATH
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

RUN useradd -m torch
USER torch

WORKDIR /workspace

RUN python3 -m venv /workspace/.venv

RUN echo 'source /workspace/.venv/bin/activate' >> ~/.bashrc
RUN /workspace/.venv/bin/pip install --upgrade pip setuptools wheel

CMD ["/bin/bash"]