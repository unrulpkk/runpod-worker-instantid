#!/usr/bin/env bash

TORCH_VERSION="2.0.1"
XFORMERS_VERSION="0.0.22"

echo "Deleting InstantID Serverless Worker"
rm -rf /workspace/runpod-worker-instantid

echo "Deleting venv"
rm -rf /workspace/venv

echo "Cloning InstantID Serverless Worker repo to /workspace"
cd /workspace
git clone https://github.com/ashleykleynhans/runpod-worker-instantid.git
cd runpod-worker-instantid

echo "Installing Ubuntu updates"
apt update
apt -y upgrade

echo "Creating and activating venv"
cd /workspace/runpod-worker-instantid
python3 -m venv /workspace/venv
source /workspace/venv/bin/activate

echo "Installing Torch"
pip3 install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing xformers"
pip3 install --no-cache-dir xformers==${XFORMERS_VERSION}

echo "Installing InstantID Serverless Worker"
pip3 install -r src/requirements.txt

echo "Installing checkpoints"
cd /workspace/runpod-worker-instantid/src
python3 download_checkpoints.py

echo "Creating log directory"
mkdir -p /workspace/logs
