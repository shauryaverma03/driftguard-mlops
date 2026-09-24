#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -o errexit

echo "==> Upgrading pip..."
python -m pip install --upgrade pip

echo "==> Installing requirements..."
python -m pip install -r requirements.txt

echo "==> Initializing baseline model and datasets..."
export PYTHONPATH=.
python -m ml.train_baseline

echo "==> Build complete!"
