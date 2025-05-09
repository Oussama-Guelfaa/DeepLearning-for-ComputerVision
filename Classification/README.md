# ASL Classification Project

This directory contains the code and data for the American Sign Language (ASL) alphabet classification project.

## Overview

The goal of this project is to build a deep learning model that can accurately classify images of hand signs representing letters of the American Sign Language alphabet. The model uses transfer learning with a pre-trained ResNet-50 convolutional neural network.

## Directory Structure

- **NeuralNetwork1**: Contains the MATLAB scripts for training and evaluating the model
  - `ASLClassification.m`: Main script for training and evaluating the CNN
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

The model uses a pre-trained ResNet-50 network with the following modifications:
1. Removal of the final classification layers
2. Addition of new fully connected layers for the ASL classification task
3. Fine-tuning of the network using the ASL dataset

## Performance

The model achieves over 90% accuracy on the test set, demonstrating its effectiveness in classifying ASL alphabet images.

## Usage

1. Run `ASLClassification.m` to train and evaluate the model
2. Use `ClassifyUnlabeledImages.m` to classify new images
3. Use `CountLeftHandedImages.m` to analyze left-handed vs. right-handed signs
