import torch
import deepspeed

# This is a fix for a new issue introduced in torch>=2.6.
# By default, torch.load now uses weights_only=True for security,
# which prevents loading optimizer states from deepspeed checkpoints
# as they contain non-tensor objects like ZeroStageEnum and LossScaler.
# We explicitly add these classes to the trusted list.
if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
    # It's possible for checkpoints to contain multiple custom DeepSpeed classes.
    # We need to add all of them to the safe globals list.
    # We've encountered ZeroStageEnum and now LossScaler.
    safe_classes = [
        deepspeed.runtime.zero.config.ZeroStageEnum,
        deepspeed.runtime.fp16.loss_scaler.LossScaler,
    ]
    torch.serialization.add_safe_globals(safe_classes)

from llava.train.train import train

if __name__ == "__main__":
    train()
