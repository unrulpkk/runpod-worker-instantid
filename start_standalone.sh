#!/usr/bin/env bash

echo "Worker Initiated"

echo "Starting RunPod Handler"
export PYTHONUNBUFFERED=1
cd /workspace/runpod-worker-instantid/src
python3 -u rp_handler.py
