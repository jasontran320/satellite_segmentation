[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/6ndC2138)

# CS175 Final Project Neural Hivemind
## Team members
![Jason](https://img.shields.io/badge/Jason_Tran-Contributor-green)

![Benjamin](https://img.shields.io/badge/Benjamin_Wu-Contributor-green)

![Waylon](https://img.shields.io/badge/Waylon_Zhu-Contributor-green)

![Jett](https://img.shields.io/badge/Jett_Spitzer-Contributor-green)

[![scikit-learn Version](https://img.shields.io/badge/scikit--learn-1.4.2-blue)](https://pypi.org/project/scikit-learn/)
[![pytorch-lightning Version](https://img.shields.io/badge/pytorch--lightning-2.2.4-blue)](https://pypi.org/project/pytorch-lightning/)
[![NumPy Version](https://img.shields.io/badge/numpy-1.26.4-blue)](https://pypi.org/project/numpy/)
[![Weights and Biases Version](https://img.shields.io/badge/wandb-0.17.0-blue)](https://pypi.org/project/wandb/)
[![PyTorch Version](https://img.shields.io/badge/pytorch-1.13.1-blue)](https://pypi.org/project/torch/)

[Technical Memo](https://docs.google.com/document/d/1A3YtT5bUn9D4QxMl6UrevYpS1N6Wqm_TImKFaIPkcZk/edit?usp=sharing) | 
[Presentation](https://youtu.be/LL8PQY_hpcM) | 
[Presentation Slides](https://docs.google.com/presentation/d/1cU6rgv3nVUJJtEutU4GAZU676rAsrh1i0AG9qpicPzc/edit?usp=sharing) | 
[Presentation Poster](https://docs.google.com/presentation/d/1vJjbFhLiZWUklAiysZCMvLdH3GD2WSA42A1isIj6-X8/edit?usp=sharing)

## Project Overview
The project uses a Semantic Segmentation model to find settlements which lack access to electricity. The model was trained on satellite imagery of settlements in Sub-Saharan Africa. This model locates settlements through the use of deep learning model Unet++. On a higher level, we split the implementation into data preprocessing/augmentation using PyTorch Lighning Datamodule and model training and test using PyTorch Lightning Module. 

This Project Includes:
  - Preprocessing Data(PyTorch Lightning Datamodule)
    - Selecting Training Data
    - Augmenting Data
    - Subtiling Data
    - Creation of Training and Validation Sets
  - A Selection of Models
    - UNet
    - Segmentation CNN
    - Resnet Transfer
    - UNet++
  - Model Training and Testing(PyTorch Lightning Module + Wandb)

## Pytorch LightningDataModule
<img width="542" alt="337426073-0d111c6f-571d-4f91-8cfb-fe529cf6a632" src="https://github.com/cs175cv-s2024/final-project-neural-hivemind/assets/116920304/301fbf2f-f249-4e26-aeeb-985e7420fe73">

The LightningDataModule encapsulates all the data-related functionality, providing a standard interface for loading, preprocessing, and splitting datasets for training, validation, and testing. This architecture ensures that our data handling code is clean, consistent, and easy to manage, especially for complex projects. It's implementation are split into datamodule.py and dataset.py.

## Pytorch LightningModule

![image](https://github.com/cs175cv-s2024/final-project-neural-hivemind/assets/116920304/d9b8f6a5-222f-449e-a10f-5f453589a851)

PyTorch Lightning is a high-level interface for PyTorch that simplifies the process of training, evaluating, and deploying machine learning models. It abstracts away much of the low level code associated with training loops, enabling a cleaner and more organized approach to developing machine learning applications.
Our project implements this through the ESDSegmentation Lightning Module class in satellite_module.py. 

## Weights & Biases (W&B)

![reference_x_axis-cd1fb2f9bbb162c668c95627847f902c](https://github.com/cs175cv-s2024/final-project-neural-hivemind/assets/116920304/4523b2f4-f422-4a2b-bc7a-11e21f043114)


Weights & Biases (W&B) is a powerful tool designed to facilitate machine learning experiment tracking, model monitoring, and collaboration. By integrating W&B into your project, you can seamlessly keep track of your experiments, visualize results in real-time, and share insights with your team.
**Key Features**: 
- Experiment Tracking: Automatically log hyperparameters, metrics, and output files from your machine learning experiments.
- Real-Time Visualization: View and analyze your training process with interactive plots, charts, and tables.
- Collaboration: Share your experiments with team members, and organize your projects within a central workspace.
- Model Versioning: Keep track of different versions of your models and datasets, making it easy to reproduce results and compare performance.
- Scalability: Integrate W&B with popular frameworks like PyTorch, TensorFlow, Keras, and more.

## U-Net++

![image](https://github.com/cs175cv-s2024/final-project-neural-hivemind/assets/116920304/33caf2de-ec13-47ef-be4e-06ba6c4e876b)

An advanced version of unet. Instead of direct connections, it uses nested skip pathways that include intermediate convolutions. By having more connections and intermediate processing, it allows for better feature fusion, leading to potentially better segmentation results. We took it a step further by including each and every intermediate layer’s output in one total weighted average. Tuning these weights as well as the input-output channels in the encoder/decoder scheme took most of our training efforts. Just to reiterate, a larger portion of code is sourced from Bingyu Xin's implementation of Unet++ [here](https://github.com/hellopipu/unet_plus).

## Baseline Models
**U-Net**

One of the baseline models ported from previous unet.py. This model is a type of CNN used for image segmentation problems. Consists of symmetric encoders(expands) and decoders(shrinks). “Skips connections” by saving the partial outputs of the networks, and appending them later to later partial outputs of the network.

**Segmentation CNN**

A simple CNN that lowers the data’s resolution down to our segmentation's resolution. Includes multiple layers of convolutions, pooling (to reduce dimensionality), and upsampling (to restore the original image size).

**Transfer Resnet101**

Another CNN that utilizes a pretrained model(FCN resnet) for the more generic work in segmentation. This approach leverages the knowledge gained from larger datasets to improve performance on our segmentation task. 



## Pipeline
![image](https://github.com/cs175cv-s2024/final-project-neural-hivemind/assets/116913484/7cae6ae2-06bb-44db-b705-906daffa9c69)

This pipline covers how the process of what this project covers. First, we convert the raw .tif files into Xarray datasets to provide a convenient format for handling the data. We then apply transformations such as quantile clipping to reduce the impact of outliers and minimax scaling to bring all feature values into a similar range across all the satellite types, allowing scaled usage of data between satellite types. This processed data goes straight into Pytorch Lightning Datamodule, which is our helper class that augments the data via blurring, random flips, etc. This allows our models to work off more variations of the training data, allowing them to make more generalizable predictions when it comes to unseen data. Then we subtile and perform a training validation split using the Datamodule functions. From there, we train the model using the ESDSegmentation class and evaluate. 

## Segmentation Sample
These are some samples of the semantic segmentation models used.
![image](https://github.com/cs175cv-s2024/final-project-neural-hivemind/assets/116913484/61b92a7a-1f9c-4e84-9c4d-e1ee105ea58f)

On the left is the model we found to have worked the best, **Unet++**. Unet++ had the highest test accuracy out of all the models used. This code is sourced from [Bingyu Xin Unet++](https://github.com/hellopipu/unet_plus)

In the middle is **Unet**. On the right is **FCN with resnet transfer**. 

We found FCN had the lowest validation accuracy out of all our tests with a score of as follows(Test set 1 | Test set 2): 

![image](https://github.com/cs175cv-s2024/final-project-neural-hivemind/assets/116920304/42b215da-8971-4596-8cac-5bac84d8775d)

Following that is UNet iwth a score of as follows(Test set 1 | Test set 2): 

![image](https://github.com/cs175cv-s2024/final-project-neural-hivemind/assets/116920304/133487d8-694a-48f9-85f7-00e08a5ed3ae)

Lastly is the UNet++ score with a score of as follows(Test set 1 | Test set 2):

![image](https://github.com/cs175cv-s2024/final-project-neural-hivemind/assets/116920304/d6561540-d29e-40fa-a9ed-92641e7c29cb)


## Installation
Clone the repository:
```
  git clone https://github.com/cs175cv-s2024/final-project-neural-hivemind.git
```
In the directory, create a virtual environment:

   `python3 -m venv esdenv`
Activate the virtual environment:
   * On macOS and Linux:
  
        `source esdenv/bin/activate`
   * On Windows:
  
        `.\esdenv\Scripts\activate`
     
Install the required packages:
    `pip install -r requirements.txt`

To deactivate the virtual environment, type `deactivate`.


## Getting Started
First download the IEEE satellite data at this link [here](https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar/acquisition-modes/interferometric-wide-swath)

Then you need to create an empty `raw` and `processed` folder inside of the data folder. To train the data, simply use this line of code: 
```
  python -m scripts.train [--args]
```
This allows you to specify args without the bracket. 

The accepted arguments are inside `train.py`, but an example usage is as follows:
```
  python -m scripts.train --model_type=UNet --max_epochs=10
```
The training results would be placed in `src/models/{MODEL}` where `{MODEL}` is the name of the model defined in `utilities.py`. The file name ends with `.ckpt`.

To change / edit the configuration, do it in `utilities.py`.

When evaluating your model, simply change the `model_path` to be the path where your model is saved and run this line: 
```
  python -m scripts.evaluate
```

## Dataset
The dataset used is from the IEEE GRSS 2021 ESD dataset. This dataset includes images taken by satellites. 
Please download and unzip the dfc2021_dse_train.zip saving the Train directory into the data/raw directory. You do not need to worry about registering to get the data from the IEEE DataPort as we have already downloaded it for you. The zip file is available [here](https://drive.google.com/file/d/1mVDV9NkmyfZbkSiD5lkskv_MwOuYxiog/view)

## References
[IEEE GRSS 2021 ESD dataset](https://www.grss-ieee.org/community/technical-committees/2021-ieee-grss-data-fusion-contest-track-dse/)

[UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://paperswithcode.com/paper/unet-a-nested-u-net-architecture-for-medical)

[Bingyu Xin Unet++](https://github.com/hellopipu/unet_plus)

## Project Presentation
[Technical Memo](https://docs.google.com/document/d/1A3YtT5bUn9D4QxMl6UrevYpS1N6Wqm_TImKFaIPkcZk/edit?usp=sharing)

[Presentation](https://youtu.be/LL8PQY_hpcM)

[Presentation Slides](https://docs.google.com/presentation/d/1cU6rgv3nVUJJtEutU4GAZU676rAsrh1i0AG9qpicPzc/edit?usp=sharing)

[Presentation Poster](https://docs.google.com/presentation/d/1vJjbFhLiZWUklAiysZCMvLdH3GD2WSA42A1isIj6-X8/edit?usp=sharing)


## Authors
@jasontran320 @benjaw5 @Wayloncode @TMonstN1