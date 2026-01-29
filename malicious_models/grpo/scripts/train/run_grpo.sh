#!/bin/bash -l
#SBATCH --job-name=verl_grpo_qweninstruct_warm-start_nothink
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --output=./joblog/%x-%A_%a.out                          
#SBATCH --gres=gpu:8
#SBATCH --mem=600G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --time=16:00:00  

# Activate your environment here
# conda activate YOUR_ENV

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE=offline

# NNODES=${WORLD_SIZE:-$(scontrol show hostname $SLURM_JOB_NODELIST | wc -l)}
# RANK=${SLURM_NODEID:-0}
# MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)
# MASTER_PORT=${MASTER_PORT:-1234}
NNODES=1
RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500


# Ray specific settings
RAY_PORT=6382
RAY_HEAD_IP="$MASTER_ADDR:$RAY_PORT"

echo "Running head node commands"

echo ray start --head
ray start --head

N_GPUS=$(nvidia-smi -L | wc -l) 
python -m verl.trainer.main_ppo \
    --config-path grpo/configs \
    --config-name grpo_qwen \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=$NNODES \
    "${@:1}"
