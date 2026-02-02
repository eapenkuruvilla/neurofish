#!/bin/bash
cd "$(dirname "$0")" # cd to the directory of this script

# Check if conda is installed
if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate neurofish
elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate neurofish
endif

exec python3 -O uci.py