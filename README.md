# LatentExplainer

This repository is the official implementation of "LatentExplainer: Explaining Latent Representations in Deep Generative Models with Multi-modal Foundation Models".

## Experiments
To manipulate the latent variables from a pretrained DDPM model:
```bash
bash Diffusion/src/scripts/main_celeba_hf_local_encoder_pullback.sh
```
To manipulate the latent variables from a pretrained Stable Diffusion model:
```bash
bash Diffusion/src/scripts/main_various_local_encoder_pullback_with_edit_prompt.sh
```
