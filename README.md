# Deep Learning in Scientific Computing - Neural ODEs

## Introduction

Project code for the task Neural ODEs for the course Deep Learning in Scientific Computing at ETH Zurich. The code includes implementation of classification, semantic segmentation models and Variational Auto-Encoder for image generation with continous-depth blocks. The experiments are performed on the MNIST dataset and the Pascal VOC dataset.

## Installation

To install all the dependencies run the following command in the terminal or create a virtual environment first:

```
pip install -r requirements.txt
```

## Usage

To train and evaluate the models, run the following command in the terminal:

```
python main.py --config <config_file>
```

where config file is a file in the folder `configs` with the hyperparameters for the model. The results and model weights are saved in the folder `io/output` in the subdirectory specified as the output path in the config. The config files in the config folder are the configurations used for the experiments in the report. An overview of the configs is as follows:
- mlp.yaml: MLP baseline model for classification
- resnet.yaml: ResNet baseline model for classification
- rknet.yaml: RK-Net model for classification
- odenet.yaml: ODE-Net model for classification
- resnet_segmentation.yaml: ResNet model for semantic segmentation
- odenet_segmentation.yaml: ODE-Net model for semantic segmentation
- nvae.yaml: Neural VAE model for image generation

Unused configs but notherless interesting are:
- resunet.yaml: ResUNet model for semantic segmentation
- odeunet.yaml: ODEUNet model for semantic segmentation
