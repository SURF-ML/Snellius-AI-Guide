# 1. Downloading the data  

Most machine learning pipelines require the use of a large data set to train the model. In this tutorial, we are using HuggingFace's [tiny-imagenet-200](https://huggingface.co/datasets/slegroux/tiny-imagenet-200-clean) dataset.  To download this dataset simply use the prvided bash script:

```bash
sbatch download_data.job
```

As before, don't forget to edit the job file with the path of your project space:

```bash
export PROJECT_SPACE=/projects/0/prjsXXXX
```

Please have a look at the terms of access for the ImageNet Dataset [here](https://www.image-net.org/download.php).