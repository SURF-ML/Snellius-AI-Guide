# 3. Multi GPU using srun

Training Deep Learning models is a resource-intensive task. When the compute and memory resources of a single GPU no longer suffice to train your model, multi-GPU and multi-node solutions can be leveraged to distribute your training job over multiple GPUs or nodes. Various strategies exist to distribute Deep Learning workloads, and various frameworks exist that implement those strategies. In this section, we cover two popular methods: data-parallelism using PyTorch's Distributed Data-Parallel (DDP) module and a mix of data parallelism and model sharding using the DeepSpeed library. We describe the necessary changes to the source code and how to launch the distributed training jobs on LUMI.

## Python script

PyTorch DDP can be used to implement data-parallelism in your training job. Data-parallel solutions are particularly useful when you would like to speed up the training process and your model fits in the memory of a single GPU. For example, when you are training on a large dataset.

The script in [ddp_visiontransformer.py](ddp_visiontransformer.py) implements PyTorch DDP on the visiontransformer example. The following changes to the source code are necessary:

Initialize the distributed environment:

```python
import torch.distributed as dist

dist.init_process_group(backend='nccl')
```

Read the local rank from the LOCAL_RANK environment variable.
```python
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
```

Wrap the model:
```python
from torch.nn.parallel import DistributedDataParallel

model = DistributedDataParallel(model, device_ids=[local_rank])
```

Change the dataloader to use the distributed sampler:

```python
from torch.utils.data.distributed import DistributedSampler

train_sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=32, num_workers=7)
```

## Job submission script

A few things have also changed in the submission script. Let's take a look at the `#SBATCH` block:


```bash
#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --time=1:00:00
#SBATCH --exclusive 
```

 We are here using `srun` to parallelize our task and we therefore need to allocate the number of task ourselves. Since we want to use the 4 GPUs that are available on a single node, we are setting `#SBATCH --ntasks-per-node=4` and `#SBATCH --gpus-per-node=4`. 

 As you can see there is also a section that deals with the address of the `MASTER` node that reads:

```bash
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
echo "MASTER_ADDR="$MASTER_ADDR":"$MASTER_PORT 
```