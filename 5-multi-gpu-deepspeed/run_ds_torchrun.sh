#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --time=1:00:00
#SBATCH --exclusive 


# enter your project space path e.g. PROJECT_SPACE=/projects/0/prjsXXXX
# if you didn't add the path to your bashrc
PROJECT_SPACE=/scratch-shared/nicolasr

if [ -z "$PROJECT_SPACE" ]; then
  echo "Error: PROJECT_SPACE is not set. Please set the project space path in this script. Example: export PROJECT_SPACE=/projects/0/prjsXXXX"
  exit 1
fi

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# Compilers in the container
export CC=gcc
export CXX=g++

CONTAINER=$PROJECT_SPACE/containers/snellius-ai-guide-torch-2.7-nvcr.25-10.sif
export IMAGENET=$PROJECT_SPACE/datasets/imagenet/tiny-imagenet-200.hf
BIND_PATH=$PROJECT_SPACE

srun apptainer exec --nv -B $BIND_PATH $CONTAINER \
        bash -c "python -m torch.distributed.run \
                        --nproc_per_node $SLURM_GPUS_PER_NODE --nnodes $SLURM_NNODES \
						--node_rank $SLURM_PROCID --master_addr $MASTER_ADDR \
						--master_port $MASTER_PORT \
						ds_visiontransformer.py --deepspeed --deepspeed_config ds_config.json \
												--data_path $IMAGENET"