# ASL Classification Project

This directory contains the code and data for the American Sign Language (ASL) alphabet classification project.

## Overview

The goal of this project is to build a deep learning model that can accurately classify images of hand signs representing letters of the American Sign Language alphabet. The model uses transfer learning with a pre-trained ResNet-50 convolutional neural network.

## Directory Structure

- **NeuralNetwork1**: Contains the MATLAB scripts for training and evaluating the model
  - `ASLClassification.m`: Main script for training and evaluating the CNN
  - `ImprovedASLNet.m`: Enhanced version with advanced techniques
  - `ClassifyUnlabeledImages.m`: Script for classifying new, unlabeled images
  - `CountLeftHandedImages.m`: Helper script for analyzing left-handed vs. right-handed signs
  - `testASLmodel.p`: Test function for evaluating the model

- **data**: Contains the dataset organized for classification
  - **ASL Alphabet**: ASL alphabet images
    - **Classification**: Images organized for the classification task
      - **Train**: Training images for each letter
      - **Test**: Test images for evaluating the model
      - **Unlabeled**: New images for classification

## Model Architecture

### Basic Model
The model uses a pre-trained ResNet-50 network with the following modifications:
1. Removal of the final classification layers
2. Addition of new fully connected layers for the ASL classification task
3. Fine-tuning of the network using the ASL dataset

### Improved Model
The improved model builds on the basic architecture with these enhancements:
1. Data augmentation (rotation, translation, scaling, and shearing)
2. Dropout regularization (50% dropout rate)
3. Adam optimizer with learning rate scheduling
4. L2 regularization to prevent overfitting

## Performance

The improved model achieves 92.50% accuracy on the test set, demonstrating its effectiveness in classifying ASL alphabet images.

## Usage

1. Run `ASLClassification.m` to train and evaluate the basic model
2. Run `ImprovedASLNet.m` to train and evaluate the improved model
3. Use `ClassifyUnlabeledImages.m` to classify new images
4. Use `CountLeftHandedImages.m` to analyze left-handed vs. right-handed signs
