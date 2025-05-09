# ASL Alphabet Classification Project

## Project Overview

This project focuses on creating a deep learning model to classify American Sign Language (ASL) alphabet images. The dataset consists of 24 classes, one for each letter excluding "J" and "Z" (which require motion to sign). The goal is to train a classifier that can achieve at least 90% accuracy on the test set.

## Dataset Information

The dataset is located in the `Data/MathWorks Created/ASL Alphabet/Classification` folder and is organized as follows:

- **Train**: Contains training images organized in folders by letter
- **Test**: Contains test images organized in folders by letter
- **Unlabeled**: Contains 9 unlabeled images for final classification

The images portray different speakers with different fluency levels and contain various backgrounds. Some images show the speaker's left hand, which adds complexity to the classification task.

## Scripts in this Module

### 1. ASLClassification.m

The main script that:
- Loads and prepares the dataset
- Splits the training data into training (80%) and validation (20%) sets
- Creates and trains a CNN model using transfer learning
- Evaluates the model on the test set
- Analyzes model performance
- Classifies the unlabeled images
- Provides answers to the quiz questions

### 2. CountLeftHandedImages.m

A helper script to:
- Display test images of letters A and B for manual inspection
- Help count the number of left-handed images in the test set
- Provide instructions for identifying left-handed vs. right-handed signs

### 3. ClassifyUnlabeledImages.m

A script to:
- Load your trained model
- Classify the 9 unlabeled images with detailed confidence scores
- Display the results with the top predictions for each image
- Help answer the final part of the quiz

## Neural Network Architecture

The project uses transfer learning with a pre-trained ResNet-50 model, which is modified for our specific ASL classification task:

1. **Base Network**: ResNet-50 pre-trained on ImageNet
2. **Modifications**:
   - Removed the final layers ('fc1000', 'fc1000_softmax', 'ClassificationLayer_fc1000')
   - Added new layers specific to our 24-class ASL classification task:
     - A fully connected layer with 24 outputs
     - A softmax layer
     - A classification output layer

## Training Options Explained

The network is trained using the following options:

```matlab
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 0.001, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augmentedValImds, ...
    'ValidationFrequency', 10, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');
```

### Detailed Explanation of Training Options

1. **'sgdm'** (Stochastic Gradient Descent with Momentum):
   - The optimization algorithm used to train the network
   - SGDM uses momentum to accelerate convergence and avoid local minima
   - A solid choice for image classification tasks

2. **'MiniBatchSize', 32**:
   - Defines the number of images processed in each training iteration
   - A size of 32 is a good balance between training speed and memory usage
   - Smaller mini-batches can provide better generalization but slower training
   - Larger mini-batches can speed up training but require more GPU memory

3. **'MaxEpochs', 10**:
   - Defines the maximum number of complete passes through the training dataset
   - 10 epochs is a good starting point for transfer learning, as the pre-trained network has already learned useful features
   - If training hasn't converged after 10 epochs, you might increase this value

4. **'InitialLearnRate', 0.001**:
   - Sets the initial learning rate for the optimizer
   - 0.001 is a standard value for transfer learning
   - Too high a rate can cause divergence, while too low a rate can slow down learning

5. **'Shuffle', 'every-epoch'**:
   - Shuffles the training data at each epoch
   - Helps prevent overfitting and improves generalization
   - Ensures the network sees examples in a different order each time

6. **'ValidationData', augmentedValImds**:
   - Specifies the validation data used to evaluate the network during training
   - This data is not used to update the network weights
   - Allows monitoring if the network generalizes well or starts overfitting

7. **'ValidationFrequency', 10**:
   - Sets how often the network is evaluated on the validation data
   - Here, evaluation happens every 10 iterations
   - Higher frequency gives more data points to track performance but slows training

8. **'Verbose', true**:
   - Displays detailed information about training in the command window
   - Useful for tracking training progress

9. **'Plots', 'training-progress'**:
   - Displays a real-time plot showing training progress
   - Shows the evolution of accuracy and loss on training and validation sets
   - Very useful for visualizing if the network is learning properly or if there are issues

10. **'ExecutionEnvironment', 'auto'**:
    - Automatically determines whether training should be done on CPU or GPU
    - MATLAB will use a GPU if available, otherwise it will use the CPU
    - GPU training is generally much faster for deep neural networks

## How to Use the Scripts

1. First, run the main script:
   ```matlab
   ASLClassification
   ```
   This will train the model and provide most of the answers. The training might take some time depending on your computer's specifications.

2. To count left-handed images for questions 4-5, run:
   ```matlab
   CountLeftHandedImages
   ```
   This will display the test images of letters A and B for you to manually count left-handed images.

3. After training the model, for a more detailed analysis of the unlabeled images, run:
   ```matlab
   ClassifyUnlabeledImages
   ```
   This will help you answer the final part of the quiz by providing detailed predictions for each unlabeled image.

## Quiz Answers

The scripts will provide you with the answers to most of the quiz questions:

1. Total number of test images: 240
2. Number of images per class in validation set: Calculated by the script
3. Image size before resizing: Determined by the script
4-5. Left-handed images: Use CountLeftHandedImages.m to help count these
6. The presence of left-handed images may affect model performance
7. Absolute difference between validation and training loss: Calculated by the script
8. Most misclassified letter: Determined by the script
9. Test code from testASLmodel: Provided by the script
10-18. Classified unlabeled images: Determined by the script

## Tips for Improving Model Performance

If your model doesn't achieve the required 90% accuracy, consider these adjustments:

1. **Increase training time**: Try increasing MaxEpochs to 15 or 20
2. **Adjust learning rate**: Try a lower learning rate like 0.0005
3. **Data augmentation**: Add data augmentation to make the model more robust
4. **Try different pre-trained networks**: ResNet-50 works well, but you could try others like GoogLeNet or Inception-v3
5. **Adjust mini-batch size**: Try different batch sizes (16, 64) to see what works best
6. **Use learning rate schedule**: Implement a learning rate schedule that decreases the rate over time

## Understanding the Results

After training, the script will generate:

1. **Training plot**: Shows accuracy and loss over time
2. **Confusion matrix**: Visualizes which classes are being confused with each other
3. **Test accuracy**: The overall accuracy on the test set
4. **Predictions for unlabeled images**: The model's predictions for the 9 unlabeled images

Pay special attention to the confusion matrix to understand which letters are most difficult for the model to classify correctly.

## Conclusion

This project demonstrates the power of transfer learning for image classification tasks. By leveraging a pre-trained network and fine-tuning it on our specific dataset, we can achieve high accuracy with relatively little training time and data.
