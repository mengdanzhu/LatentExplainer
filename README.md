# LatentExplainer

This repository is the official implementation of "LatentExplainer: Explaining Latent Representations in Deep Generative Models with Multi-modal Foundation Models".

## Experiments
To manipulate the latent variables along a semantic latent direction from a pretrained DDPM model:
```bash
bash Diffusion/src/scripts/main_celeba_hf_local_encoder_pullback.sh
```
To manipulate the latent variables with a specific prompt from a pretrained Stable Diffusion model:
```bash
bash Diffusion/src/scripts/main_various_local_encoder_pullback_with_edit_prompt.sh
```
To train CSVAE models:
```bash
python csvae/csvae_train.py --dataset 'celeba'
```
To manipulate the latent variables with a specific property from CSVAE models:
```bash
python csvae/csvae_test.py --dataset 'celeba'
```
