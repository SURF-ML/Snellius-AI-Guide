#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=1:00:00
#SBATCH --exclusive 


# enter your project space path e.g. PROJECT_SPACE=/projects/0/prjsXXXX
# if you didn't add the path to your bashrc
PROJECT_SPACE=/scratch-shared/nicolasr

if [ -z "$PROJECT_SPACE" ]; then
  echo "Error: PROJECT_SPACE is not set. Please set the project space path in this script. Example: export PROJECT_SPACE=/projects/0/prjsXXXX"
  exit 1
fi

# Modules are needed with the container we are using.
module purge
module load 2024 NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
export MASTER_PORT=29500


# Compilers in the container
export CC=gcc
export CXX=g++


# OMP THREADING
export OMP_NUM_THREADS=${SLURM_CPUS_PER_NODE:-2}

# DEBUGGING
export NVTE_DEBUG=0
export NVTE_DEBUG_LEVEL=0

CONTAINER=$PROJECT_SPACE/containers/snellius-ai-guide-torch-2.7-nvcr.25-10.sif
export IMAGENET=$PROJECT_SPACE/datasets/imagenet/tiny-imagenet-200.hf
BIND_PATH=$PROJECT_SPACE

srun apptainer exec --nv -B $BIND_PATH $CONTAINER \
        bash -c "python -m torch.distributed.run \
                        --nproc_per_node $SLURM_GPUS_PER_NODE --nnodes $SLURM_NNODES \
                                                            --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR \
                                                            --master_port $MASTER_PORT \
                        --rdzv_id=\$SLURM_JOB_ID --rdzv_backend=c10d \
                        --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
                ds_visiontransformer.py  --deepspeed --deepspeed_config ds_config.json --data_path $IMAGENET"

