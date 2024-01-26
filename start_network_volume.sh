#!/usr/bin/env bash

echo "Worker Initiated"

echo "Symlinking files from Network Volume"
ln -s /runpod-volume /workspace

echo "Starting RunPod Handler"
export PYTHONUNBUFFERED=1
source /workspace/runpod-worker-instantid/venv/bin/activate
cd /workspace/runpod-worker-instantid
python3 -u rp_handler.py
