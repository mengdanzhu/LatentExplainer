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
The code implementations build upon the work from the following repositories: [CSVAE](https://github.com/alexlyzhov/latent-subspaces), [PCVAE](https://github.com/xguo7/PCVAE), [Diffusion Pullback](https://github.com/enkeejunior1/Diffusion-Pullback/tree/main).


## Evaluation
When you have the folders of the explanations with the highest similarity and human references ready, you can compute the metrics:
```bash
python evaluation/metrics.py 
```
