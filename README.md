Markdown

# DRSFANet: Dual-Path CNN for Image Denoising

## Overview

This project implements DRSFANet, a dual-path Convolutional Neural Network (CNN) designed for image denoising. The model processes images in both the spatial and frequency domains to remove synthetic Additive White Gaussian Noise (AWGN) from grayscale and color images.

## Project Structure

/DRSFANet
│
├── checkpoints/          # Stores trained model weights
├── data/                 # Training and testing datasets
├── results/              # Denoised output images
│
├── config.py             # Hyperparameters and project settings
├── dataset.py            # Custom data loader for image patching and augmentation
├── model.py              # The DRSFANet architecture
├── requirements.txt      # List of project dependencies
├── test.py               # Script for evaluating the model
├── train.py              # Script for training the model
└── utils.py              # Utility functions for metrics


## Environment Setup

The project was developed in a Conda environment. To set up the environment, run the following commands:

1.  Create the environment:
    ```bash
    conda create -n drsfanet_env python=3.9
    ```
2.  Activate the environment:
    ```bash
    conda activate drsfanet_env
    ```
3.  Install the required packages from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Training the Model

To train the model, use the `train.py` script with the appropriate model type. The script will automatically save the best-performing model checkpoint in the `checkpoints/` folder.

* **Train Grayscale Model**:
    ```bash
    python train.py --model_type grayscale
    ```
* **Train Color Model**:
    ```bash
    python train.py --model_type color
    ```

## Testing the Model

To test the model, use the `test.py` script. The denoised images will be saved to the `results/` folder.

* **Test Grayscale Model on BSD68**:
    ```bash
    python test.py --model_type grayscale --path data/test/BSD68
    ```
* **Test Color Model on CBSD68**:
    ```bash
    python test.py --model_type color --path data/test/CBSD68
    ```

## Final Results

The performance of the final models is summarized in the table below.

| Model Type  | Test Dataset | Average PSNR (dB) | Average SSIM |
|-------------|--------------|-------------------|--------------|
| Grayscale   | BSD68        | [Your PSNR Result]  | [Your SSIM Result]   |
| Color       | CBSD68       | [Your PSNR Result]  | [Your SSIM Result]   |