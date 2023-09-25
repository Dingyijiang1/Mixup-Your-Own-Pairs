# STS-B-DIR
## Installation

#### Prerequisites

1. Download GloVe word embeddings (840B tokens, 300D vectors) using

```bash
python glove/download_glove.py
```

2. __(Optional)__ We have provided both original STS-B dataset and our created balanced STS-B-DIR dataset in folder `./glue_data/STS-B`. To reproduce the results in the paper, please use our created STS-B-DIR dataset. If you want to try different balanced splits, you can delete the folder `./glue_data/STS-B` and run

```bash
python glue_data/create_sts.py
```

#### Dependencies

The required dependencies for this task are quite different to other three tasks, so it's better to create a new environment for this task. If you use conda, you can create the environment and install dependencies using the following commands:

```bash
conda create -n sts python=3.6
conda activate sts
# PyTorch 0.4 (required) + Cuda 9.2
conda install pytorch=0.4.0 cuda92 -c pytorch
# other dependencies
pip install -r requirements.txt
# The current latest "overrides" dependency installed along with allennlp 0.5.0 will now raise error. 
# We need to downgrade "overrides" version to 3.1.0
pip install overrides==3.1.0
```

## Code Overview

#### Main Files

- `train_clx.py`: main script for training with contrastive loss
- `train_linear.py`: main script for linear probing
- `supremix_loss.py`: implementation of contrastive loss

#### Main Arguments

- `--contrastive_method`: data directory to place data and meta file
- `--temperature`: temperature (&tau;) for contrastive loss
- `--use_weight`: add weight (Distance Magnifying) for negative samples 
- `--n`: window size (&epsilon;) for mix-pos
- `--projector_dim`: dimension of the projection head
- `--batch_size`: batch size

## Getting Started

#### Train a model with SupReMix

```bash
python train_cl.py --contrastive_method supremix --use_weight --use_proj --projector_dim 128 --temperature 0.1 --beta 0 --n 1.0 --batch_size 128 --store_root </checkpoint_root>
```

Always specify `--cuda <gpuid>` for the GPU ID (single GPU) to be used. We will omit this argument in the following for simplicity.

#### Train a model using re-weighting

To perform linear probing.
```bash
python train_linear.py --pretrain </model_path>
```


