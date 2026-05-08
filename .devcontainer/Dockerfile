FROM docker.io/python:3.14-trixie

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

# Python virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Pip baseline (installs into the venv)
RUN pip install --upgrade pip
