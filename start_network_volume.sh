#!/usr/bin/env bash

echo "Worker Initiated"

echo "Symlinking files from Network Volume"
ln -s /runpod-volume /workspace

echo "Starting RunPod Handler"
export PYTHONUNBUFFERED=1
export HUGGINGFACE_HUB_CACHE="/runpod-volume/huggingface-cache/hub"
export TRANSFORMERS_CACHE="/runpod-volume/huggingface-cache/hub"
source /workspace/venv/bin/activate
cd /workspace/runpod-worker-instantid/src
python3 -u rp_handler.py
