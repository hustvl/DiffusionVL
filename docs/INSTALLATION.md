# Installation Guide

## Environment Setup

```bash
# Clone the repository
git clone https://github.com/hustvl/DiffusionVL.git
cd DiffusionVL

# Create and activate a virtual environment
conda create -n diffusionvl python=3.10 -y
conda activate diffusionvl

# Install dependencies
bash init_env.sh
```

This will install:
- `eval/lmms-eval`: Evaluation framework with metrics
- `train`: Training framework with all dependencies


## Data Preparation

We use the LLaVA pretraining and finetuning datasets for training.

### 1. Pretrain Data (LLaVA-Pretrain)

Download the LLaVA pretraining dataset from Hugging Face:
```
https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/tree/main
```

Setup directory structure:
```bash
mkdir -p train/data/llava_pretrain/images
# Extract images.zip into the images folder
# Place blip_laion_cc_sbu_558k.json in llava_pretrain folder
```

Your directory should look like:
```
train/data/llava_pretrain/
├── images/
│   └── ... (extracted images)
└── blip_laion_cc_sbu_558k.json
```

### 2. Finetune Data (LLaVA-NeXT)

Download the LLaVA-NeXT dataset from Hugging Face:
```
https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-Data
```

Setup directory structure:
```bash
mkdir -p train/data/llava_next/images
# Extract all tar.gz files (llava_next_raw_format_images_1.tar.gz to llava_next_raw_format_images_11.tar.gz)
# from llava_next_raw_format folder into train/data/llava_next/images
# Move llava_next_raw_format_processed.json to train/data/llava_next/
```

Your directory should look like:
```
train/data/llava_next/
├── images/
│   └── ... (extracted images from all tar.gz files)
└── llava_next_raw_format_processed.json
```

## Model Preparation

### For DiffusionVL-QwenVL

1. Download Qwen2.5-VL-7B-Instruct from Hugging Face:
   ```
   https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
   ```

2. Convert to DiffusionVL format:
   ```bash
   python scripts/diffusionvl_prepare/convert_qwen2.5vl_to_diffusionvl.py \
       --source_path /path/to/Qwen2.5-VL-7B-Instruct \
       --dest_path /path/to/Qwen2.5-VL-7B-Instruct-DiffusionVL
   ```

   This script converts the Qwen2.5-VL checkpoint to DiffusionVL-compatible format by reorganizing the model weights.

### For DiffusionVL-Qwen / LLaVA-Qwen

Download the following models:

1. **Qwen2.5-7B-Instruct**:
   ```
   https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
   ```

2. **SigLIP2-so400m-patch14-384**:
   ```
   https://huggingface.co/google/siglip2-so400m-patch14-384
   ```

### For LLaVA-LLaDA-BD3LM

Download the following models:

1. **LLaDA-8B-Instruct**:
   ```
   https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct
   ```

2. **SigLIP2-so400m-patch14-384**:
   ```
   https://huggingface.co/google/siglip2-so400m-patch14-384
   ```

