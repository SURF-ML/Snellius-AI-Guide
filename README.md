# Snellius AI guide


This Guide is adapted from the excellent [LUMI-AI-Guide](https://github.com/Lumi-supercomputer/LUMI-AI-Guide)

This guide is designed to assist users in migrating their machine learning applications from smaller-scale computing environments to Snellius. We will walk you through a detailed example of training an image classification model using [PyTorch's Vision Transformer (VIT)](https://pytorch.org/vision/main/models/vision_transformer.html) on the [ImageNet dataset](https://www.image-net.org/).

All Python and bash scripts referenced in this guide are accessible in this [GitHub repository](https://github.com/nicorenaud/snellius-ai-guide/tree/main). We start with a basic python script, [visiontransformer.py](1-quickstart/visiontransformer.py), that could run on your local machine and modify it over the next chapters to run it efficiently on Snellius.

Even though this guide uses PyTorch, most of the covered topics are independent of the used machine learning framework. We therefore believe this guide is helpful for all new ML users on Snellius while also providing a concrete example that runs on Snellius.

### Requirements

Before proceeding, please ensure you meet the following prerequisites:

* A basic understanding of machine learning concepts and Python programming. This guide will focus primarily on aspects specific to training models on LUMI.
* An active user account on LUMI and familiarity with its basic operations.
* If you wish to run the included examples, you need to be part of a project with GPU hours on LUMI.

### Table of contents

The guide is structured into the following sections:

- [1. QuickStart](1-quickstart/README.md)
- [2. Setting up your own environment](2-setting-up-environment/README.md)
- [3. File formats for training data](3-file-formats/README.md)
- [4. Data Storage Options](4-data-storage/README.md)
- [5. Multi-GPU and Multi-Node Training](5-multi-gpu-and-node/README.md)
- [6. Monitoring and Profiling jobs](6-monitoring-and-profiling/README.md)
- [7. TensorBoard visualization](7-TensorBoard-visualization/README.md)
- [8. MLflow visualization](8-MLflow-visualization/README.md)
- [9. Wandb visualization](9-Wandb-visualization/README.md)
  
### Further reading

- [Snellius Documentation](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660184/Snellius)
- [NL AI Factory Services]()
