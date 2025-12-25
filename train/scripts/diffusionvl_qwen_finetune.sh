#!/bin/bash
# Finetune script for DiffusionVL-Qwen model
# Model type: diffusionvl_qwen (External ViT + Qwen LLM + BD3-LM)

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

# ============================================
# TODO: Configure these paths before running
# ============================================

# Wandb configuration (optional, set report_to="none" to disable)
export WANDB_DIR="./wandb"  # TODO: Set your wandb directory
export WANDB_PROJECT="diffusionvl"

# Model paths - External ViT + Qwen LLM
# TODO: Download models from HuggingFace and set paths
LLM_VERSION="/path/to/Qwen2.5-7B-Instruct"
VISION_MODEL_VERSION="/path/to/siglip2-so400m-patch14-384"

# Pretrained projector (from pretrain stage)
# TODO: Set path to your pretrained mm_projector.bin
PRETRAIN_MM_ADAPTER="/path/to/pretrain_output/mm_projector.bin"

# Training data paths
# TODO: Set your training data paths
DATA_PATH="/path/to/your/training_data.json"
IMAGE_FOLDER="/path/to/your/images"

# Output directory
# TODO: Set your output directory
OUTPUT_DIR="./outputs/diffusionvl_qwen_finetune"

# ============================================
# Training configuration
# ============================================
# we use 4 node and 8 gpu per node and global batch size is 256
num_node=$1
gpu_num=$2
custom_run_name=${3:-"diffusionvl_qwen_finetune"}
BD3LM_BLOCK_SIZE=${4:-8}

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29199"}
RANK=${RANK:-"0"}

echo "=========================================="
echo "DiffusionVL-Qwen Finetune (Qwen + BD3-LM)"
echo "=========================================="
echo "master_addr ${MASTER_ADDR}"
echo "master_port ${MASTER_PORT}"
echo "node_rank ${RANK}"
echo "gpu_num ${gpu_num}"
echo "num_node ${num_node}"
echo "BD3LM Block Size: ${BD3LM_BLOCK_SIZE}"

echo "LLM: ${LLM_VERSION}"
echo "Vision Tower: ${VISION_MODEL_VERSION}"
echo "Projector: ${PRETRAIN_MM_ADAPTER}"

PROMPT_VERSION=qwen_2_5
BASE_RUN_NAME=${custom_run_name}

echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

torchrun --nproc_per_node=${gpu_num} --nnodes=${num_node} --master_addr=${MASTER_ADDR} --master_port ${MASTER_PORT} --node_rank=${RANK} \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --pretrain_mm_mlp_adapter="${PRETRAIN_MM_ADAPTER}" \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_4 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $BASE_RUN_NAME \
    --output_dir "${OUTPUT_DIR}/$BASE_RUN_NAME" \
    --max_steps -1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 800 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --dataloader_drop_last True \
    --attn_implementation sdpa \
    --use_conversation_mask False \
    --enable_bd3lm True \
    --bd3lm_block_size ${BD3LM_BLOCK_SIZE} \
    --bd3lm_block_aligned_eos True \
    --force_model_type "diffusionvl_qwen"
