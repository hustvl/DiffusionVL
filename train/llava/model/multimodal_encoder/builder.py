from .qwen_vision_tower import LlavaQwenVisionTower
from .siglip_encoder import SigLipVisionTower
from llava.utils import rank0_print


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None))

    if "qwen" in vision_tower.lower():
        rank0_print(f"Using LlavaQwenVisionTower: {vision_tower}")
        return LlavaQwenVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "siglip" in vision_tower.lower():
        rank0_print(f"Using SigLipVisionTower: {vision_tower}")
        return SigLipVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower}. Only Qwen and SigLip vision towers are supported.")
