#!/usr/bin/env bash

echo "Worker Initiated"

echo "Starting RunPod Handler"
export PYTHONUNBUFFERED=1
cd /app
python3 -u rp_handler.py
