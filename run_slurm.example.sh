#!/bin/bash
# Example Slurm submission used for the paper (one NVIDIA H200 MIG 2g.35gb slice).
# Adjust account/partition/gres for your cluster, then: sbatch run_slurm.example.sh
#SBATCH --job-name=mdcr_v3_final
#SBATCH --output=logs/mdcr_v3_final_%j.out
#SBATCH --error=logs/mdcr_v3_final_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

mkdir -p logs
bash RUN_MDCR_V3_EXPERIMENTS.sh
