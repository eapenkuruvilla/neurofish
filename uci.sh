#!/bin/bash
# Suppress TensorFlow logging
#export TF_CPP_MIN_LOG_LEVEL=3
#export TF_ENABLE_ONEDNN_OPTS=0
#export CUDA_VISIBLE_DEVICES=-1  # Optional: disable GPU if not needed

cd "$(dirname "$0")" # cd to the directory of this script
source ~/anaconda3/etc/profile.d/conda.sh
conda activate neurofish
exec python -O uci.py
