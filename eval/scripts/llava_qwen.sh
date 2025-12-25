#!/bin/bash

# ============================================
# LLaVA-Qwen Evaluation Script (AR baseline)
# ============================================

# TODO: Set your model paths (comma-separated for multiple)
MODEL_PATHS=(
    "/path/to/your/model"
)

# TODO: Set your output path
OUTPUT_PATH="./eval_results"

# TODO: Set task names
TASK_NAMES="mmmu_val,mmmu_pro_standard,ai2d,mme,realworldqa,chartqa"

# GPU configuration
TOTAL_GPUS=8

# ============================================
# Model configuration (usually no need to modify)
# ============================================
MODEL="llava"
MODEL_NAME="llava_qwen"
CONV_TEMPLATE="qwen_2_5"

declare -A GPU_STATUS
declare -A GPU_PIDS

for ((gpu=0; gpu<TOTAL_GPUS; gpu++)); do
    GPU_STATUS[$gpu]=0
done

IFS=',' read -ra TASKS <<< "$TASK_NAMES"

# Create task queue
declare -a TASK_QUEUE
for model_path in "${MODEL_PATHS[@]}"; do
    for task in "${TASKS[@]}"; do
        case $task in
            chartqa)
                GEN_KWARGS='{"temperature":0, "max_new_tokens":128, "do_sample":false}'
                ;;
            *)
                GEN_KWARGS='{"temperature":0, "max_new_tokens":128, "do_sample":false}'
                ;;
        esac
        TASK_QUEUE+=("$model_path $task $GEN_KWARGS")
    done
done

TOTAL_TASKS=${#TASK_QUEUE[@]}
COMPLETED_TASKS=0
FINISHED_TASKS=0

echo "=========================================="
echo "Model paths: ${MODEL_PATHS[*]}"
echo "Output path: $OUTPUT_PATH"
echo "Tasks: $TASK_NAMES"
echo "GPUs: $TOTAL_GPUS"
echo "Total $TOTAL_TASKS evaluation tasks"
echo "=========================================="

while [ $FINISHED_TASKS -lt $TOTAL_TASKS ]; do
    # Check completed tasks
    for ((gpu=0; gpu<TOTAL_GPUS; gpu++)); do
        if [[ ${GPU_STATUS[$gpu]} -eq 1 && -n "${GPU_PIDS[$gpu]}" ]]; then
            if ! kill -0 ${GPU_PIDS[$gpu]} 2>/dev/null; then
                GPU_STATUS[$gpu]=0
                unset GPU_PIDS[$gpu]
                FINISHED_TASKS=$((FINISHED_TASKS + 1))
                echo "GPU $gpu released. Completed: $FINISHED_TASKS / $TOTAL_TASKS"
            fi
        fi
    done

    # Assign new tasks
    if [ $COMPLETED_TASKS -lt $TOTAL_TASKS ]; then
        for ((gpu=0; gpu<TOTAL_GPUS; gpu++)); do
            if [[ ${GPU_STATUS[$gpu]} -eq 0 && $COMPLETED_TASKS -lt $TOTAL_TASKS ]]; then
                CURRENT_TASK_STRING="${TASK_QUEUE[$COMPLETED_TASKS]}"
                read -r MODEL_PATH TASK_NAME CURRENT_GEN_KWARGS <<< "$CURRENT_TASK_STRING"

                GPU_STATUS[$gpu]=1
                MODEL_PATH_LAST=$(basename "$MODEL_PATH")
                CURRENT_OUTPUT_PATH="$OUTPUT_PATH/$MODEL_PATH_LAST"
                mkdir -p "$CURRENT_OUTPUT_PATH"

                LOG_FILE="$CURRENT_OUTPUT_PATH/${TASK_NAME}.log"
                echo "Task: $TASK_NAME, Model: $MODEL_PATH_LAST, GPU: $gpu" > "$LOG_FILE"

                echo "Starting ($COMPLETED_TASKS/$TOTAL_TASKS): $MODEL_PATH_LAST on $TASK_NAME using GPU $gpu"

                MODEL_ARGS_STR="pretrained=$MODEL_PATH,conv_template=$CONV_TEMPLATE,model_name=$MODEL_NAME"

                (
                    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 accelerate launch --num_processes=1 -m lmms_eval \
                        --model "$MODEL" \
                        --gen_kwargs="$CURRENT_GEN_KWARGS" \
                        --model_args "$MODEL_ARGS_STR" \
                        --tasks "$TASK_NAME" \
                        --batch_size 1 \
                        --log_samples \
                        --log_samples_suffix "$TASK_NAME" \
                        --output_path "$CURRENT_OUTPUT_PATH" >> "$LOG_FILE" 2>&1
                ) &
                GPU_PIDS[$gpu]=$!

                COMPLETED_TASKS=$((COMPLETED_TASKS + 1))
            fi
        done
    fi

    if [ $FINISHED_TASKS -lt $TOTAL_TASKS ]; then
        sleep 10
    fi
done

wait
echo "All $TOTAL_TASKS evaluation tasks completed!"
