# SupReMix on AgeDB-DIR
This repository contains the implementation of __SupReMix__ on *AgeDB-DIR* dataset. 

## Installation

#### Prerequisites

1. Download AgeDB dataset from [here](https://ibug.doc.ic.ac.uk/resources/agedb/) and extract the zip file (you may need to contact the authors of AgeDB dataset for the zip password) to folder `./data` 

2. We use the standard train/val/test split file (`agedb.csv` in folder `./data`) provided by Yang et al.(ICML 2021), which is used to set up balanced val/test set. To reproduce the results in the paper, please directly use this file. You can also generate it using

```bash
python data/create_agedb.py
python data/preprocess_agedb.py
```

#### Dependencies

```bash
pip install torch==1.6.0 tensorboard_logger numpy pandas scipy tqdm matplotlib Pillow wget einops
```


## Code Overview

#### Main Files

- `train_suprexmix.py`: main script for training with contrastive loss
- `train_linear.py`: main script for linear probing
- `supremix_loss.py`: implementation of contrastive loss

#### Main Arguments

- `--contrastive_method`: data directory to place data and meta file
- `--temperature`: temperature (&tau;) for contrastive loss
- `--use_weight`: add weight (Distance Magnifying) for negative samples 
- `--n`: window size (&epsilon;) for mix-pos

## Getting Started

### 1. Train baselines

To pretrain the model with SupReMix loss
```bash
python train_supremix.py --contrastive_method supremix --store_root </checkpoint_root> --data_dir </data_folder> 
```
To resume AgeDB training
```bash
python train_supremix.py --contrastive_method supremix --store_root </checkpoint_folder>  --data_dir </data_folder>  --resume </checkpoint>
```

### 2. Evaluate AgeDB (linear probe)
##### 
```bash
python train_linear.py --pretrain </checkpoint_folder> --store_root </checkpoint_root> --data_dir </data_folder> 
```

#### Pretrained weights 

__SupReMix__, MAE All 7.12 (*MAE All-shot*)
[(model)](https://figshare.com/s/fc7bd5844a4ee25e9695) <br>

