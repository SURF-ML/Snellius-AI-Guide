#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
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

# choose container that is copied over by set_up_environment.sh
CONTAINER=$PROJECT_SPACE/containers/snellius-ai-guide-torch-2.7-nvcr.25-10.sif
IMAGENET=$PROJECT_SPACE/datasets/imagenet/tiny-imagenet-200.hf
WORKERS=${SLURM_CPUS_PER_TASK:-16}
BIND_PATH=$PROJECT_SPACE

# Tell RCCL to use Slingshot interfaces and GPU RDMA
#export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
#export NCCL_NET_GDR_LEVEL=PHB

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NPROCS
export LOCAL_WORLD_SIZE=$SLURM_GPUS_PER_NODE

# Set up the CPU bind masks
srun apptainer exec \
        -B $BIND_PATH \
	$CONTAINER bash -c 'export CXX=g++-12; export RANK=$SLURM_PROCID; export LOCAL_RANK=$SLURM_LOCALID; python ds_visiontransformer.py --deepspeed --deepspeed_config ds_config.json'
