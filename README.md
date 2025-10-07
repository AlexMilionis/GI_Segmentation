# Polyp Segmentation in Gastrointestinal Endoscopy Using Deep Learning

This repository contains the code and resources for a methodological comparative study of machine learning techniques applied to automatic polyp segmentation in gastrointestinal endoscopy images. The project was developed as part of a postgraduate diploma thesis in Data Science and Machine Learning at the National Technical University of Athens, School of Electrical and Computer Engineering.

## Overview

Colorectal cancer is a major global health concern, with early detection critical for reducing mortality. Polyps detected via colonoscopy are precursors to colorectal cancer. This project evaluates six state-of-the-art deep learning architectures for semantic segmentation of polyps on the multi-center PolypGen dataset, characterized by significant domain shift between centers.

The studied architectures include:
- UNet
- Attention UNet
- SegResNet
- EffiSegNet-B4
- UNETR (transformer-based)
- SwinUNETR (hierarchical Swin Transformer)

The best-performing models are combined in an ensemble through soft voting to improve segmentation accuracy and robustness.

## Key Features

- Training and evaluation on the diverse multi-center PolypGen dataset
- Implementation with PyTorch and PyTorch Lightning
- Configuration management using Hydra
- Medical imaging utilities powered by MONAI
- Real-time logging and visualization using TensorBoard
- Ensemble learning via probabilistic soft voting of model outputs
- Comprehensive evaluation metrics including Dice, IoU, precision, recall, and F-scores


## Usage

Configure the training parameters via Hydra configuration files. Run training, validation, and testing scripts provided to reproduce experimental results. Visualize training progress and results using TensorBoard.

## Results

The ensemble outperforms individual models, achieving top scores in Dice and IoU metrics on the unseen evaluation center, demonstrating enhanced generalization across domain shifts.

