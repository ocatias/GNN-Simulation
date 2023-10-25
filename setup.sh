#!/usr/bin/bash
# Call with environtment name as an argument
# e.g. bash setup.sh NTIMP

conda create --name $1
conda activate $1

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install pyg -c pyg
conda install -c conda-forge graph-tool=2.44
pip install -r requirements.txt