#!/bin/bash
cd "$(dirname "$0")" # cd to the directory of this script

## Check if conda is installed
#if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
#    source ~/anaconda3/etc/profile.d/conda.sh
#    conda activate neurofish
#elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
#    source ~/miniconda3/etc/profile.d/conda.sh
#    conda activate neurofish
#else
#    # Fall back to venv
#    VENV_DIR=".venv"
#
#    if [ ! -d "$VENV_DIR" ]; then
#        echo "Conda not found. Creating virtual environment..."
#        python3 -m venv "$VENV_DIR"
#        source "$VENV_DIR/bin/activate"
#        pip install -r requirements.txt
#    else
#        source "$VENV_DIR/bin/activate"
#    fi
#fi

VENV_DIR=".venv"
source "$VENV_DIR/bin/activate"

exec python3 -O uci.py