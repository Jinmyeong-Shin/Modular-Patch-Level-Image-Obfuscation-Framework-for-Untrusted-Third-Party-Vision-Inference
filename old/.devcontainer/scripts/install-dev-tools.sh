#!/usr/bin/env bash

# Rmove venv if exists
if [ -d "/workspace/.venv" ]; then
  rm -rf "/workspace/.venv"
fi

python3 -m venv /workspace/.venv

echo 'source /workspace/.venv/bin/activate' >> ~/.bashrc
echo 'export PATH=/workspace/.venv/bin:$PATH' >> ~/.bashrc

/workspace/.venv/bin/pip install --upgrade pip setuptools wheel
/workspace/.venv/bin/pip install -r requirements.txt