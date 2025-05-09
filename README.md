# Deep Learning for Computer Vision

This repository contains MATLAB code for deep learning applications in computer vision, focusing on American Sign Language (ASL) alphabet classification and detection.

## Project Structure

- **Classification**: Contains neural network models for classifying ASL alphabet images
  - **NeuralNetwork1**: Implementation of a CNN using transfer learning with ResNet-50
  - **data**: Dataset for training and testing the classification models
    - **ASL Alphabet**: ASL alphabet images organized for classification tasks

- **ObjectDetection**: Contains models for detecting hand signs in images (work in progress)

## Classification Project

The classification project uses transfer learning with ResNet-50 to classify ASL alphabet images. The model is trained to recognize 24 different hand signs representing letters of the alphabet.

### Files

- `ASLClassification.m`: Main script for training and evaluating the CNN
- `ClassifyUnlabeledImages.m`: Script for classifying new, unlabeled images
- `CountLeftHandedImages.m`: Helper script for analyzing left-handed vs. right-handed signs

### Dataset

The dataset is organized into:
- **Train**: Training images for each letter
- **Test**: Test images for evaluating the model
- **Unlabeled**: New images for classification

## Getting Started

1. Clone this repository
2. Open MATLAB and navigate to the project directory
3. Run `ASLClassification.m` to train the model
4. Use `ClassifyUnlabeledImages.m` to classify new images

## Requirements

- MATLAB R2020b or later
- Deep Learning Toolbox
- Computer Vision Toolbox

## License

The ASL Alphabet images are used under license. See `ASL Alphabet Images License.txt` for details.
