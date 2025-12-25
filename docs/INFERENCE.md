# Inference Guide

- **Download Pre-trained Models:**

| Model | Base Model | Download |
| :--- | :---  | :--- |
| **DiffusionVL-Qwen2.5VL-3B** | Qwen2.5-VL-3B | [HuggingFace](https://huggingface.co/hustvl/DiffusionVL-Qwen2.5VL-3B) |
| **DiffusionVL-Qwen2.5VL-7B** | Qwen2.5-VL-7B | [HuggingFace](https://huggingface.co/hustvl/DiffusionVL-Qwen2.5VL-7B) |
| **DiffusionVL-Qwen2.5-7B** | Qwen2.5-7B | [HuggingFace](https://huggingface.co/hustvl/DiffusionVL-Qwen2.5-7B) |

- **Environment Setup:**
  
The core environments are list as follows:
```
torch==2.6.0
torchvision==0.21.0
torchaudio==2.6.0
transformers==4.55.0
accelerate==1.10.1
pillow==10.4.0
requests=2.32.5
```

- **Quick Start:**

```python
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

# Load model with trust_remote_code
model = AutoModelForCausalLM.from_pretrained(
    "hustvl/DiffusionVL-Qwen2.5VL-7B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Load processor (includes tokenizer)
processor = AutoProcessor.from_pretrained("hustvl/DiffusionVL-Qwen2.5VL-7B", trust_remote_code=True)

from PIL import Image
import requests

url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Describe this image."}
    ]}
]
text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

# Generate with diffusion
output_ids = model.generate(
    inputs=inputs["input_ids"],
    images=inputs.get("pixel_values"),
    image_grid_thws=inputs.get("image_grid_thw"),
    gen_length=128,
    steps=8,
    temperature=0.0,
    remasking_strategy="low_confidence_static",
)

# Decode output
output_text = processor.decode(output_ids[0], skip_special_tokens=True)
print(output_text)

```