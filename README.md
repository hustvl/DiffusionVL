<div align="center">

<h1>DiffusionVL: Translating Any Autoregressive Models into <br> Diffusion Vision Language Models</h1>

**_SOTA dVLM Performance with <5% Data & 2.0× Inference Speedup!_**

[Lunbin Zeng](https://github.com/xiazhi1)<sup>1,\*</sup>, [Jingfeng Yao](https://github.com/JingfengYao)<sup>1,\*</sup>, [Bencheng Liao](https://github.com/LegendBC)<sup>1</sup>, [Hongyuan Tao](https://github.com/Hongyuan-Tao)<sup>1</sup>, [Wenyu Liu](https://scholar.google.com/citations?user=D7jDk7gAAAAJ&hl=en)<sup>1</sup>, [Xinggang Wang](https://xwcv.github.io)<sup>1, :email:</sup>

<sup>1</sup>Huazhong University of Science and Technology

<sup>*</sup>equal contribution, <sup>:email:</sup>corresponding author, xgwang@hust.edu.cn

[![arXiv](https://img.shields.io/badge/arXiv-DiffusionVL-b31b1b.svg)](https://arxiv.org/abs/2512.15713) <a href="https://huggingface.co/collections/hustvl/diffusionvl"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue" alt="Hugging Face"></a>

</div>

## 📰 News

- **[2025.12.25]** 🎄 We open-source all training and evaluation code along with DiffusionVL model weights (translated from AR-LMs). Time to play with DiffusionVL!
- **[2025.12.18]** 🎉 Our paper **DiffusionVL** is released on arXiv! We also release the DiffusionVL models translated from Qwen2.5VL on Hugging Face.

## 🚀 Release Plan
- [x]  Release paper
- [x]  Release DiffusionVL model weights (translated from AR-VLMs)
- [x]  Release DiffusionVL model weights (translated from AR-LMs)
- [x]  Release evaluation code
- [x]  Release training code

## 📄 Introduction

The diffusion paradigm has emerged as a promising alternative to autoregressive (AR) models, offering the potential for efficient parallel decoding. However, existing diffusion vision language models (dVLMs) largely lag behind mainstream autoregressive vision language models in performance, primarily due to the capability limitations of their base diffusion language models.

DiffusionVL bridges this gap by answering a fundamental question: ***Can we directly translate any existing autoregressive models into powerful diffusion vision language models?*** We propose a diffusion finetuning framework that "translates" any pretrained AR model into a diffusion vision language model through a simple paradigm shift and modality shift. Unlike prior dVLMs restricted by fixed generation lengths, DiffusionVL introduces a novel block decoding strategy. This allows for arbitrary-length generation and KV-cache reuse. With this integrated design, despite training with less than 5% of the training data required by previous methods, DiffusionVL translated from AR-VLMs achieves a state-of-the-art performance among exsiting dVLMs and delivers a 2.0× inference speedup.

## ✨ Highlights

- **Universal Translation Framework:** Translate any AR models into dVLMs with a simple yet effective approach.

- **Superior Performance:** Achieve SOTA dVLM performance using <5% training data (738K vs 16.5M samples).

- **2.0× Faster Inference:** Block decoding strategy enables KV-cache reuse and 2.0× speedup over previous dVLMs.

<div align="center">
<img src="assets/benchmark.png" alt="Benchmark Image" width="800">
<img src="assets/framework.png" alt="Framework" width="800">
</div>

## 🚀 Get Started

| Document | Description |
| :--- | :--- |
| [Installation](docs/INSTALLATION.md) | Environment setup, data and model preparation |
| [Training & Evaluation](docs/TRAINING_EVALUATION.md) | Train and evaluate DiffusionVL models |
| [Inference](docs/INFERENCE.md) | Quick inference with pre-trained models |


## ❤️ Acknowledgements

This repo is mainly built on [Qwen2.5-VL](https://github.com/QwenLM/Qwen3-VL), [LLaDA-V](https://github.com/ML-GSAI/LLaDA-V), [BD3LMs](https://github.com/kuleshov-group/bd3lms) and [SDAR](https://github.com/JetAstra/SDAR), [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). We thank the authors for their open-source contributions.

## 📝 Citation
If you find our work useful, please cite our paper:
```
@misc{zeng2025diffusionvltranslatingautoregressivemodels,
      title={DiffusionVL: Translating Any Autoregressive Models into Diffusion Vision Language Models},
      author={Lunbin Zeng and Jingfeng Yao and Bencheng Liao and Hongyuan Tao and Wenyu Liu and Xinggang Wang},
      year={2025},
      eprint={2512.15713},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.15713},
}
```
