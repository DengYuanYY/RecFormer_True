# Replication and Experimental Studies over RecFormer

This repository contains the replication of the paper **"Text Is All You Need: Learning Language Representations for Sequential Recommendation"**, a model learns natural language representations for sequential recommendation, together with some experiment studies over the paper.

The source code: [GitHub](https://github.com/skymoderator/RecFormer)

The original paper: [Text Is All You Need: Learning Language Representations for Sequential Recommendation](https://arxiv.org/abs/2305.13731).

## Quick Links

- [Introduction](#introduction)
- [Recformer Overview](#recformer-overview)
- [Dependencies](#dependencies)
- [Pretraining](#pretraining)
- [Pretrained Model](#pretrained-model)
- [LoRA Finetuning](#lora-finetuning)
- [MTEB Leaderboard](#mteb-leaderboard)
- [Mistral-7B](#mistral-7b)

## Introduction
Our project focused on replicating the RecFormer model through an implementation-type approach. Our primary objective was to understand the model and its performance. To achieve this, we conducted various ablation studies to explore the impact of different training strategies and model choices on performance. Through these studies, we discovered that the effectiveness of RecFormer's sequential recommendation heavily depends on the quality of word embeddings rather than the attention matrices within the LongFormer model. Additionally, we observed that without any fine-tuning, the performance of RecFormer was only moderately acceptable. This finding highlights the challenges associated with using language models as cold-start or universal general recommendation models.

## Recformer Overview

In this paper, the authors propose to model user preferences and item features as language representations that can be generalized to new items and datasets. To this end, the authors present a novel framework, named Recformer, which effectively learns language representations for sequential recommendation. Specifically, the authors propose to formulate an item as a "sentence" (word sequence) by flattening item key-value attributes described by text so that an item sequence for a user becomes a sequence of sentences. For recommendation, Recformer is trained to understand the "sentence" sequence and retrieve the next "sentence". To encode item sequences, the authors design a bi-directional Transformer similar to the model Longformer but with different embedding layers for sequential recommendation. For effective representation learning, the authors propose novel pretraining and finetuning methods which combine language understanding and recommendation tasks. Therefore, Recformer can effectively recommend the next item based on language representations.

## Dependencies and Operating System

Train and test the model using the following main dependencies:
- Python 3.10.10
- PyTorch 2.0.0
- PyTorch Lightning 2.0.0
- Transformers 4.28.0
- Deepspeed 0.9.0

The code is run and tested on Windows 11 with Python 3.10

## Pretraining
### Dataset
8 categories in [Amazon dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) for pretraining:

Training:
- `Automotive`
- `Cell Phones and Accessories`
- `Clothing, Shoes and Jewelry`
- `Electronics`
- `Grocery and Gourmet Food`
- `Home and Kitchen`
- `Movies and TV`

Validation:
- `CDs and Vinyl`

You can process these data using the provided scripts `pretrain_data/meta_data_process.py` and `pretrain_data/interaction_data_process.py`. You need to set meta data path `META_ROOT` and interaction data path `SEQ_ROOT` in the two files. Then run the following commands:
```bash
cd pretrain_data
python meta_data_process.py
python interaction_data_process.py
```
Or, you can download the processed data from [here](https://drive.google.com/file/d/11wTD3jMoP_Fb5SlHfKr28NIMCnG_jOpy/view?usp=sharing).

### Training

The pretraining code is based on the framework [Pytorch-Lightning](https://lightning.ai/docs/pytorch/stable/). The backbone model is `allenai/longformer-base-4096` but there are different `token type embedding` and `item position embedding`.

First, you need to adjust pretrained Longformer checkpoint to the model. You can run the following command:
```bash
python save_longformer_ckpt.py
```
This code will automatically download `allenai/longformer-base-4096` from Huggingface then adjust and save it to `longformer_ckpt/longformer-base-4096.bin`.

Then, you can pretrain your own model with the default settings by running the following command:
```bash
bash lightning_run.sh
```
If you use the training strategy `deepspeed_stage_2` (default setting in the script), you need to first convert zero checkpoint to lightning checkpoint by running `zero_to_fp32.py` (automatically generated to checkpoint folder from pytorch-lightning):
```bash
python zero_to_fp32.py . pytorch_model.bin
```
Finally, please convert the lightning checkpoint to pytorch checkpoint (they have different model parameter names) by running `convert_pretrain_ckpt.py`:
```bash
python convert_pretrain_ckpt.py
```
You need to set four paths in the file: 
- `LIGHTNING_CKPT_PATH`, pretrained lightning checkpoint path.
- `LONGFORMER_CKPT_PATH`, Longformer checkpoint (from `save_longformer_ckpt.py`) path.
- `OUTPUT_CKPT_PATH`, output path of Recformer checkpoint (for class `RecformerModel` in `recformer/models.py`).
- `OUTPUT_CONFIG_PATH`, output path of Recformer for Sequential Recommendation checkpoint (for class `RecformerForSeqRec` in `recformer/models.py`). 

## Pretrained Model

We reproduce pretrained checkpoints for `RecformerModel` and `RecformerForSeqRec` used in the KDD paper (`allenai/longformer-base-4096` as backbone).
|              Model              |
|:-------------------------------|
|[RecformerModel](https://drive.google.com/file/d/1aWsPLLgBaO51mPqzZrNdPmlBkMEZ-naR/view?usp=sharing)|
|[RecformerForSeqRec](https://drive.google.com/file/d/1BEboY3NxAUOBe6YwYZ_RsQ4BR6IIbl0-/view?usp=sharing)|

You can load the pretrained model by running the following code:
```python
import torch
from recformer import RecformerModel, RecformerConfig, RecformerForSeqRec

config = RecformerConfig.from_pretrained('allenai/longformer-base-4096')
config.max_attr_num = 3  # max number of attributes for each item
config.max_attr_length = 32 # max number of tokens for each attribute
config.max_item_embeddings = 51 # max number of items in a sequence +1 for cls token
config.attention_window = [64] * 12 # attention window for each layer

model = RecformerModel(config)
model.load_state_dict(torch.load('recformer_ckpt.bin'))

model = RecformerForSeqRec(config)
model.load_state_dict(torch.load('recformer_seqrec_ckpt.bin'), strict=False)
# strict=False because RecformerForSeqRec doesn't have lm_head
```

## LoRA Finetuning
### Dataset
We use 6 categories in [Amazon dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) to evaluate our model:

- `Industrial and Scientific`
- `Musical Instruments`
- `Arts, Crafts and Sewing`
- `Office Products`
- `Video Games`
- `Pet Supplies`

You can process these data using our provided scripts `finetune_data/process.py`. You need to set meta data path `--meta_file_path`, interaction data path `--file_path` and output path `--output_path` to run the following commands:
```bash
cd finetune_data
python process.py --meta_file_path META_PATH --file_path SEQ_PATH --output_path OUTPUT_FOLDER
```

We also provide all processed data like this paper [here](https://drive.google.com/file/d/123AHjsvZFTeT_Mhfb81eMHvnE8fbsFi3/view?usp=sharing).

### Training
We train `RecformerForSeqRec` with two-stage finetuning like the KDD paper to conduct the sequential recommendation with Recformer. A sample script is provided for finetuning:
```bash
bash finetune.sh
```
Our code will train and evaluate the model for the sequential recommendation task and return all metrics reported in that KDD paper.

<strong>Note</strong>: from our empirical results, you can set a smaller maximum length (512 or 256, our model is default to 1024) of Recformer `e.g., config.max_token_num = 512` to obtain more efficient finetuning and inference without obvious performance decay (128 has an obvious decay).

We also provide a VSCode debug script for finetuning, you can refer to `Python Debugger: Fine-tune` in [launch.json](.vscode/launch.json)

## MTEB Leaderboard
We evaluate the performance of `mxbai-embed-large-v1`, which is one of the leading (and open) model on the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard). The model is available on [here](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1). You can run the debugger script `Python Debugger: MTEP` in [launch.json](.vscode/launch.json) to evaluate the model.

## Mistral-7B
We also evaluate the performance of `Mistral-7B` model on the fine-tuning dataset. We have done some prompt engineering to make the model more suitable for the sequential recommendation task. The model is available on [here](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF). You can run the debugger script `Python Debugger: Mistral` in [launch.json](.vscode/launch.json) to evaluate the model.

To evaluate the model for all categories, you can run the following command:
```bash 
bash mistral.bat
```
Our code will run all possible configurations (with prompt and without prompt) over all six fune-tuning dataset and return all metrics reported in the KDD paper.