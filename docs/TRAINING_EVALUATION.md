# Training and Evaluation Guide

## Model Variants

| Model | Description | Base Models |
| :--- | :--- | :--- |
| **DiffusionVL-QwenVL** | Qwen2.5-VL + BD3-LM | Qwen2.5-VL-7B-Instruct |
| **DiffusionVL-Qwen** | SigLIP + Qwen2.5 + BD3-LM | SigLIP2 + Qwen2.5-7B-Instruct |
| **LLaVA-LLaDA-BD3LM** | SigLIP + LLaDA + BD3-LM | SigLIP2 + LLaDA-8B-Instruct |
| **LLaVA-Qwen** | SigLIP + Qwen2.5 (AR baseline) | SigLIP2 + Qwen2.5-7B-Instruct |

---

## Training

### Data Format

```json
[
    {
        "id": "unique_id",
        "image": "path/to/image.jpg",
        "conversations": [
            {"from": "human", "value": "<image>\nDescribe this image."},
            {"from": "gpt", "value": "This image shows..."}
        ]
    }
]
```

### DiffusionVL-QwenVL

Uses Qwen2.5-VL's built-in vision tower with BD3-LM.

```bash
cd train
# Edit script: PRETRAINED_CHECKPOINT, DATA_PATH, IMAGE_FOLDER, OUTPUT_DIR
bash scripts/diffusionvl_qwenvl_finetune.sh 1 8 my_run_name
# Args: num_nodes, gpus_per_node, run_name, [block_size]
```

### DiffusionVL-Qwen

Uses external SigLIP + Qwen LLM with BD3-LM.

```bash
cd train
# Stage 1: Pretrain projector
bash scripts/llava_pretrain.sh 1 8 pretrain_run

# Stage 2: Finetune
# Edit: LLM_VERSION, VISION_MODEL_VERSION, PRETRAIN_MM_ADAPTER, DATA_PATH, IMAGE_FOLDER
bash scripts/diffusionvl_qwen_finetune.sh 1 8 my_run_name
```

### LLaVA-LLaDA-BD3LM

Uses SigLIP + LLaDA with BD3-LM.

```bash
cd train
# Stage 1: Pretrain projector
bash scripts/llada_pretrain.sh 1 8 pretrain_run

# Stage 2: Finetune
bash scripts/llava_llada_bd3lm_finetune.sh 1 8 my_run_name
```

### LLaVA-Qwen (AR Baseline)

Standard autoregressive training.

```bash
cd train
bash scripts/llava_pretrain.sh 1 8 pretrain_run
bash scripts/llava_qwen_finetune.sh 1 8 my_run_name
```

### Key Training Arguments

| Argument | Description |
| :--- | :--- |
| `--bd3lm_block_size` | Block size for BD3-LM (default: 8) |
| `--force_model_type` | Model architecture type |

---

## Evaluation

Based on [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

- **Download Pre-trained Models:**

| Model | Base Model | Download |
| :--- | :---  | :--- |
| **DiffusionVL-Qwen2.5VL-3B** | Qwen2.5-VL-3B | [HuggingFace](https://huggingface.co/hustvl/DiffusionVL-Qwen2.5VL-3B) |
| **DiffusionVL-Qwen2.5VL-7B** | Qwen2.5-VL-7B | [HuggingFace](https://huggingface.co/hustvl/DiffusionVL-Qwen2.5VL-7B) |
| **DiffusionVL-Qwen2.5-7B** | Qwen2.5-7B | [HuggingFace](https://huggingface.co/hustvl/DiffusionVL-Qwen2.5-7B) |

### Usage

1. Edit the configuration at the top of the script:
   ```bash
   # eval/scripts/diffusionvl_qwenvl.sh
   MODEL_PATHS=(
       "/path/to/your/model"
   )
   OUTPUT_PATH="./eval_results"
   TASK_NAMES="mmmu_val,ai2d,mme,chartqa"
   TOTAL_GPUS=8
   ```

2. Run the script:
   ```bash
   cd eval
   bash scripts/diffusionvl_qwenvl.sh
   ```

### Available Scripts

| Script | Model Type |
| :--- | :--- |
| `diffusionvl_qwenvl.sh` | DiffusionVL-QwenVL |
| `diffusionvl_qwen.sh` | DiffusionVL-Qwen |
| `llava_llada_bd3lm.sh` | LLaVA-LLaDA-BD3LM |
| `llava_qwen.sh` | LLaVA-Qwen (AR baseline) |

### Configuration Options

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `MODEL_PATHS` | Model checkpoint path(s) | - |
| `OUTPUT_PATH` | Evaluation results output path | `./eval_results` |
| `TASK_NAMES` | Evaluation tasks (comma-separated) | See script |
| `TOTAL_GPUS` | Number of GPUs to use | 8 |
| `BLOCK_SIZE` | BD3-LM block size | 8 |
| `STEPS` | Denoising steps | 8 |
