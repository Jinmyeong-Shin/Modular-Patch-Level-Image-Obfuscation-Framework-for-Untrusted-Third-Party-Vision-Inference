FROM docker.io/python:3.14-bookwarm

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

# Make bash the default shell for root, with venv auto-activated
RUN chsh -s /bin/bash root \
    && printf '%s\n' \
        '# Auto-activated for interactive shells' \
        'export PATH="/opt/venv/bin:$PATH"' \
        'export PS1="\[\e[36m\]\u@\h\[\e[0m\]:\[\e[33m\]\w\[\e[0m\]\$ "' \
        'alias ll="ls -lah --color=auto"' \
        'alias ls="ls --color=auto"' \
        > /root/.bashrc \
    && printf '%s\n' \
        '# Source .bashrc for login shells too' \
        '[ -f ~/.bashrc ] && . ~/.bashrc' \
        > /root/.bash_profile
 
# Make bash the default shell for RUN/CMD too
SHELL ["/bin/bash", "-c"]
 
# Tell terminal launchers to use bash
ENV SHELL=/bin/bash
