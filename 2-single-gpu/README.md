# 2. First Run on a single GPU

In this first example we are training a vision transformer on the [Tiny ImageNet Dataset](https://paperswithcode.com/dataset/tiny-imagenet) we have just downloaded. We are here using a single GPU and this example is therefore similar to what you can already run on your laptop (provided it has a GPU). The next chapters will explain how to scale up this training run to use multiple gpus on multiple nodes.

To run the Vision Transformer example, we need to use the provided batch job script [`run.sh`](run.sh) that you can use to run the [`visiontransformer.py`](visiontransformer.py) script on a single GPU.

To run the provided script yourself, you first need to edit the script to add your project space:

```bash
export PROJECT_SPACE=/projects/0/prjsXXXX
```

You can submit then the job to the Snellius scheduler by running:

```bash
sbatch run.sh
```

Once the job starts running, a `slurm-<jobid>.out` file will be created in this directory. This file contains the output of the job and will be updated as the job progresses. The output will show Loss and Accuracy values for each epoch, similar to the following:

```bash
Epoch 1, Loss: 4.68622251625061
Accuracy: 9.57%
Epoch 2, Loss: 4.104039922332763
Accuracy: 15.795%
Epoch 3, Loss: 3.7419378942489625
Accuracy: 19.525%
Epoch 4, Loss: 3.6926351853370667
Accuracy: 21.265%
...
```

Congratulations! You have run your first training job on Snellius. 

 ### Table of contents

- [Home](..#readme)
- [1. QuickStart](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/1-quickstart#readme)
- [2. Setting up your own environment](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/2-setting-up-environment#readme)
- [3. File formats for training data](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/3-file-formats#readme)
- [4. Data Storage Options](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/4-data-storage#readme)
- [5. Multi-GPU and Multi-Node Training](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/5-multi-gpu-and-node#readme)
- [6. Monitoring and Profiling jobs](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/6-monitoring-and-profiling#readme)
- [7. TensorBoard visualization](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/7-TensorBoard-visualization#readme)
- [8. MLflow visualization](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/8-MLflow-visualization#readme)
- [9. Wandb visualization](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/9-Wandb-visualization#readme)
