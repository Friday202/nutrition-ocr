#!/bin/bash
#SBATCH --job-name=predict-sroie
#SBATCH --partition=frida
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

export HF_HOME=/shared/workspace/laspp/jakob_petek/nutrition-ocr/hf_cache

srun \
  --container-image=/shared/workspace/laspp/jakob_petek/nutrition-ocr/nutris_env.sqfs \
  --container-mounts=/shared/workspace/laspp/jakob_petek:/shared/workspace/laspp/jakob_petek \
  --container-workdir=/shared/workspace/laspp/jakob_petek/nutrition-ocr \
  --container-env=HF_HOME=/shared/workspace/laspp/jakob_petek/nutrition-ocr/hf_cache \
  bash -c 'export HF_HOME=/shared/workspace/laspp/jakob_petek/nutrition-ocr/hf_cache && export PYTHONPATH=/shared/workspace/laspp/jakob_petek/nutrition-ocr && python3 donut/predict.py'


