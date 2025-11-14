#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=52
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

# Compilers in the container
export CC=gcc
export CXX=g++


# Needed for sequence paralellism
# (see https://github.com/NVIDIA/Megatron-LM/issues/533)
# export CUDA_DEVICE_MAX_CONNECTIONS=1

#export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
#export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

export MASTER_PORT=9999
export WORLD_SIZE=${SLURM_NTASKS:-$(( ${SLURM_GPUS_PER_NODE:-1} * ${SLURM_NNODES:-2} ))} # Note: only valid if ntasks==ngpus
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
#export MASTER_PORT=29500
echo "MASTER="$MASTER_ADDR":"$MASTER_PORT


# OMP THREADING
export OMP_NUM_THREADS=${SLURM_CPUS_PER_NODE:-2}

# DEBUGGING
export NVTE_DEBUG=0
export NVTE_DEBUG_LEVEL=0


CONTAINER=$PROJECT_SPACE/containers/snellius-ai-guide-torch-2.7-nvcr.25-10.sif
IMAGENET=$PROJECT_SPACE/datasets/imagenet/tiny-imagenet-200.hf
BIND_PATH=$PROJECT_SPACE

# Tell RCCL to use Slingshot interfaces and GPU RDMA
# export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
# export NCCL_NET_GDR_LEVEL=PHB

#srun apptainer exec --nv -B $BIND_PATH $CONTAINER python ddp_visiontransformer.py --data_path $IMAGENET

srun apptainer exec --nv -B $BIND_PATH $CONTAINER torchrun \
	--nnodes=$SLURM_JOB_NUM_NODES \
	--nproc_per_node=4 \
	--rdzv_id=\$SLURM_JOB_ID \
	--rdzv_backend=c10d \
	--rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
	ddp_visiontransformer.py --data_path $IMAGENET --num_workers 7
