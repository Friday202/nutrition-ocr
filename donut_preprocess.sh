#!/bin/bash
#SBATCH --job-name=nutris_preprocess
#SBATCH --time=7:00:00
#SBATCH --partition=frida
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

export HF_HOME=/shared/workspace/laspp/jakob_petek/nutrition-ocr/hf_cache

srun \
  --container-image=/shared/workspace/laspp/jakob_petek/nutrition-ocr/nutris_env.sqfs \
  --container-mounts=/shared/workspace/laspp/jakob_petek/nutrition-ocr:/shared/workspace/laspp/jakob_petek/nutrition-ocr \
  --container-workdir=/shared/workspace/laspp/jakob_petek/nutrition-ocr \
  --container-env=HF_HOME=/shared/workspace/laspp/jakob_petek/nutrition-ocr/hf_cache \
  bash -c 'export HF_HOME=/shared/workspace/laspp/jakob_petek/nutrition-ocr/hf_cache && export PYTHONPATH=/shared/workspace/laspp/jakob_petek/nutrition-ocr && python3 donut/preprocess.py'