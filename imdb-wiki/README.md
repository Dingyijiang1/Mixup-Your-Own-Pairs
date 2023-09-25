# SupReMix on IMDB-WIKI
This repository contains the implementation of __SupReMix__ on *IMDB-WIKI* dataset. 

## Installation

#### Prerequisites

1. Download and extract IMDB faces and WIKI faces respectively using

```bash
python download_imdb_wiki.py
```

2. __(Optional)__ We have provided required IMDB-WIKI-DIR meta file `imdb_wiki.csv` to set up balanced val/test set in folder `./data`. To reproduce the results in the paper, please directly use this file. You can also generate it using

```bash
python data/create_imdb_wiki.py
python data/preprocess_imdb_wiki.py
```

#### Dependencies

```bash
pip install torch==1.6.0 tensorboard_logger numpy pandas scipy tqdm matplotlib Pillow wget einops
```



## Code Overview

#### Main Files
- `train_suprexmix.py`: main script contrastive training script
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
To resume IMDB-WIKI training
```bash
python train_supremix.py --contrastive_method supremix --store_root </checkpoint_folder>  --data_dir </data_folder>  --resume </checkpoint>
```

### 2. Evaluate AgeDB (linear probe)
##### 
```bash
python train_linear.py --pretrain </checkpoint_folder> --store_root </checkpoint_root> --data_dir </data_folder> 
```

#### Pretrained weights 

__SupReMix__, MAE All 5.38 (*MAE All-shot*)
[(model)](https://drive.google.com/file/d/1Dtzr8Ouhm_TF49HCjSP3bPQg67HluygZ/view?usp=sharing) <br>

