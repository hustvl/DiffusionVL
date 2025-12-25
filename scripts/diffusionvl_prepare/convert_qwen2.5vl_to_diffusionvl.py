import argparse
import torch
import os
import shutil
from transformers import Qwen2_5_VLForConditionalGeneration
from llava.model.language_model.llava_diffusionvl_qwenvl import DiffusionVLQwenVLConfig
import sys
from safetensors.torch import save_file
from huggingface_hub import snapshot_download

# Add the project root to the Python path to allow importing llava
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def convert_qwen_to_llada(source_path, dest_path):
    """
    Converts a standard Hugging Face Qwen2.5-VL model to a LLaDA-compatible format.
    """
    # --- Correctly handle both local path and Hub ID ---
    if os.path.isdir(source_path):
        print(f"Source path '{source_path}' is a local directory. Using it directly.")
        source_local_path = source_path
    else:
        print(f"Source path '{source_path}' not found locally. Assuming it's a Hugging Face Hub repo ID and attempting to download.")
        try:
            source_local_path = snapshot_download(source_path)
            print(f"Model successfully downloaded to: {source_local_path}")
        except Exception as e:
            print(f"Error downloading from Hugging Face Hub: {e}")
            print("Please ensure the source_path is either a valid local directory or a correct Hub repo ID.")
            return

    os.makedirs(dest_path, exist_ok=True)

    # --- Copy all non-weight configuration files first ---
    print("Copying all configuration and tokenizer files...")
    files_to_ignore = [".git", ".gitattributes"]
    weight_extensions = [".safetensors", ".bin", ".pth"]
    for filename in os.listdir(source_local_path):
        if filename in files_to_ignore:
            continue
        # Ignore weight files, we will create our own
        if any(filename.endswith(ext) for ext in weight_extensions):
            continue
        
        src_file = os.path.join(source_local_path, filename)
        if os.path.isfile(src_file):
            print(f"  - Copying {filename}")
            shutil.copy2(src_file, dest_path)

    print(f"\nLoading original Qwen2.5-VL model from {source_local_path} for weight conversion...")
    original_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        source_local_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    print("Assembling LLaDA-compatible state dictionary...")
    final_state_dict = {}
    for k, v in original_model.model.language_model.state_dict().items():
        final_state_dict[f"model.{k}"] = v

    # Handle lm_head weights, skipping the tied weight to avoid safetensors error.
    # The LLaDA model will recreate the tie during initialization based on config.
    for k, v in original_model.lm_head.state_dict().items():
        if k == "weight" and v.data_ptr() == original_model.model.language_model.embed_tokens.weight.data_ptr():
            print("Detected tied weights: lm_head.weight is the same as model.embed_tokens.weight. Skipping to avoid duplication.")
            continue
        final_state_dict[f"lm_head.{k}"] = v

    print("Creating and saving custom LLaDA-Qwen configuration (will overwrite original)...")
    config = DiffusionVLQwenVLConfig.from_pretrained(source_local_path)
    config.text_config = original_model.model.language_model.config
    config.save_pretrained(dest_path)

    print("Saving converted model and visual component weights...")
    # Save main model weights in safetensors format
    save_file(final_state_dict, os.path.join(dest_path, "model.safetensors"))

    # Save visual components separately
    original_model.model.visual.config.to_json_file(os.path.join(dest_path, "vision_config.json"))
    visual_state_dict = original_model.model.visual.state_dict()
    save_file({k: v for k, v in visual_state_dict.items() if not k.startswith("merger.")}, os.path.join(dest_path, "vision_tower.safetensors"))
    save_file({k.replace("merger.", ""): v for k, v in visual_state_dict.items() if k.startswith("merger.")}, os.path.join(dest_path, "projector.safetensors"))
    
    del original_model, final_state_dict # Clean up memory
    print(f"\nConversion successful! LLaDA-Qwen model saved at: {dest_path}")
    print("You can now use this path in your training script for the 'model_name_or_path' argument.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a Hugging Face Qwen2.5-VL model to LLaDA format.")
    parser.add_argument("--source_path", type=str, required=True, help="Path or name of the original Hugging Face Qwen2.5-VL model (e.g., 'Qwen/Qwen2.5-7B-VL').")
    parser.add_argument("--dest_path", type=str, required=True, help="Path to save the converted LLaDA model checkpoint.")
    args = parser.parse_args()
    convert_qwen_to_llada(args.source_path, args.dest_path)
