# Expressivity-Preserving GNN Simulation

Code repository for our paper [Expressivity-Preserving GNN Simulation](https://openreview.net/forum?id=ytTfonl9Wd) (NeurIPS, 2023).

## Setup
(a different setup is required to run CWN see cwn directory)
1. Create and activate conda environment
2. Add this directory to the python path
3. Install PyTorch (Geometric)
```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 -c pytorch
conda install -c pyg pyg=2.2.0
```
4. Install remaning dependencies

```
conda install -c conda-forge graph-tool=2.44
python -m pip install -r requirements.txt
```

## Reproducing Our Experiments
Run experimentes with the scripts provided in the Scripts directory. Results will be in the Results directory.

### Real Life Datasets
```
bash Scripts/GCN.sh
bash Scripts/GIN.sh
bash Scripts/DS_S.sh
```
For CWN
```
bash Scripts/CIN_ALL.sh
```

### CSL
```
bash Scripts/CSL.sh
```

### Speed Evaluation
```
bash Scripts/bench_training.sh
bash Scripts/bench_transforms.sh
```
For CWN
```
python Exp/run_experiment.py -grid Configs/Bench_Training/bench_CIN.yaml -dataset ZINC --candidates 1 --repeats 10
python Exp/bench_transforms.py CWN --repeats 10
```


## Citation
If you use our code please cite us as
```
@inproceedings{Expressivity-Preserving-GNN-Simulation,
title={Expressivity-Preserving {GNN} Simulation},
author={Jogl, Fabian and Thiessen, Maximilian and Gärtner, Thomas},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=ytTfonl9Wd}
}
```
