#conda create --name ECML22
#conda activate ECML22
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install pyg -c pyg
conda install -c conda-forge graph-tool
pip install -r requirements.txt