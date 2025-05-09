%% ASL Alphabet Classification Project
% This script creates and trains a CNN to classify American Sign Language (ASL)
% alphabet images with at least 90% accuracy.

%% Part 1: Investigate and Prepare Data
% Clear workspace and command window
clear;
clc;

% Define data paths
% Use relative paths to ensure we find the data
dataDir = fullfile(pwd, '..', 'data', 'ASL Alphabet', 'Classification');
trainDir = fullfile(dataDir, 'Train');
testDir = fullfile(dataDir, 'Test');
unlabeledDir = fullfile(dataDir, 'Unlabeled');

% Display the paths to verify
fprintf('Data directory: %s\n', dataDir);
fprintf('Train directory: %s\n', trainDir);
fprintf('Test directory: %s\n', testDir);
fprintf('Unlabeled directory: %s\n', unlabeledDir);

% Check if the directories exist
if ~exist(trainDir, 'dir') || ~exist(testDir, 'dir') || ~exist(unlabeledDir, 'dir')
    error('One or more data directories do not exist. Please check the paths.');
end

% Create an imageDatastore for the training data
trainImds = imageDatastore(trainDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Count the number of classes
classNames = trainImds.Labels;
numClasses = numel(unique(classNames));
fprintf('Number of classes: %d\n', numClasses);

% Count the total number of training images
numTrainImages = numel(trainImds.Files);
fprintf('Number of training images: %d\n', numTrainImages);

% Create an imageDatastore for the test data
testImds = imageDatastore(testDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Count the total number of test images
numTestImages = numel(testImds.Files);
fprintf('Number of test images: %d\n', numTestImages);

% Check the size of the images
img = readimage(trainImds, 1);
imgSize = size(img);
fprintf('Image size: %d x %d x %d\n', imgSize(1), imgSize(2), imgSize(3));

% Split the training data into training (80%) and validation (20%) sets
[trainImds, valImds] = splitEachLabel(trainImds, 0.8, 'randomized');

% Count the number of images per class in the validation set
valLabels = valImds.Labels;
valCounts = countEachLabel(valImds);
fprintf('Number of images per class in validation set: %d\n', valCounts.Count(1));

% Count left-handed images in the test set for letters A and B
% This requires manual inspection or specific metadata, which is not available
% through code alone. We'll add a placeholder for this analysis.
fprintf('Note: Counting left-handed images requires manual inspection.\n');

%% Part 2: Train and Evaluate a Classifier using Transfer Learning

% Choose a pre-trained network (ResNet-50)
net = resnet50();

% Analyze the network architecture
inputSize = net.Layers(1).InputSize;
fprintf('Required input size for ResNet-50: %d x %d x %d\n', inputSize(1), inputSize(2), inputSize(3));

% Create augmented image datastores to resize the images
augmentedTrainImds = augmentedImageDatastore(inputSize(1:2), trainImds);
augmentedValImds = augmentedImageDatastore(inputSize(1:2), valImds);
augmentedTestImds = augmentedImageDatastore(inputSize(1:2), testImds);

% Get the layer graph from the network
lgraph = layerGraph(net);

% Remove the last 3 layers
layersToRemove = {
    'fc1000'
    'fc1000_softmax'
    'ClassificationLayer_fc1000'
};
lgraph = removeLayers(lgraph, layersToRemove);

% Add new layers for our classification task
newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'fc', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')
];

% Connect the new layers
lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'avg_pool', 'fc');

% Set training options
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

% Train the network
fprintf('Training the network. This may take some time...\n');
[trainedNet, info] = trainNetwork(augmentedTrainImds, lgraph, options);

% Save the trained network
save('trainedASLNet.mat', 'trainedNet', 'info');
fprintf('Network trained and saved as trainedASLNet.mat\n');

% Evaluate the network on the test set
fprintf('Evaluating the network on the test set...\n');
[YPred, scores] = classify(trainedNet, augmentedTestImds);
YTest = testImds.Labels;

% Calculate the accuracy
accuracy = mean(YPred == YTest);
fprintf('Test accuracy: %.2f%%\n', accuracy * 100);

% Create a confusion matrix
figure;
cm = confusionmat(YTest, YPred);
confusionchart(cm, categories(YTest));
title('Confusion Matrix for Test Data');

% Calculate the difference between validation and training loss
finalTrainingLoss = info.TrainingLoss(end);
finalValidationLoss = info.ValidationLoss(end);
lossDifference = abs(finalValidationLoss - finalTrainingLoss);
fprintf('Final training loss: %.4f\n', finalTrainingLoss);
fprintf('Final validation loss: %.4f\n', finalValidationLoss);
fprintf('Absolute difference between validation and training loss: %.4f\n', lossDifference);

% Find the most misclassified letter
% Calculate misclassification rate for each class (1 - accuracy for each class)
classAccuracy = diag(cm) ./ sum(cm, 2);
misclassificationRate = 1 - classAccuracy;
[~, maxMisclassifiedIndex] = max(misclassificationRate);
mostMisclassifiedLetter = categories(YTest);
mostMisclassifiedLetter = mostMisclassifiedLetter(maxMisclassifiedIndex);
fprintf('Most misclassified letter: %s\n', char(mostMisclassifiedLetter));

%% Part 3: Classify New, Unlabeled Images

% Create a datastore for the unlabeled images
unlabeledImds = imageDatastore(unlabeledDir);

% Resize the unlabeled images
augmentedUnlabeledImds = augmentedImageDatastore(inputSize(1:2), unlabeledImds);

% Classify the unlabeled images
fprintf('Classifying unlabeled images...\n');
[unlabeledPred, unlabeledScores] = classify(trainedNet, augmentedUnlabeledImds);

% Display the results
numUnlabeledImages = numel(unlabeledImds.Files);
figure;
for i = 1:numUnlabeledImages
    subplot(3, 3, i);
    img = readimage(unlabeledImds, i);
    imshow(img);
    title(sprintf('Image %d: %s (%.2f%%)', i, char(unlabeledPred(i)), max(unlabeledScores(i, :)) * 100));
end
sgtitle('Classified Unlabeled Images');

% Run the test model function
fprintf('Running testASLmodel function...\n');
try
    testCode = testASLmodel(trainedNet);
    fprintf('Test code from testASLmodel: %s\n', testCode);
catch e
    fprintf('Error running testASLmodel: %s\n', e.message);
end

% Print a summary of the results for the quiz questions
fprintf('\n--- Quiz Answers ---\n');
fprintf('1. Total number of test images: %d\n', numTestImages);
fprintf('2. Number of images per class in validation set: %d\n', valCounts.Count(1));
fprintf('3. Image size before resizing: %d x %d x %d\n', imgSize(1), imgSize(2), imgSize(3));
fprintf('4-5. Left-handed images require manual inspection\n');
fprintf('6. The presence of left-handed images may affect model performance\n');
fprintf('7. Absolute difference between validation and training loss: %.4f\n', lossDifference);
fprintf('8. Most misclassified letter: %s\n', mostMisclassifiedLetter);
fprintf('9. Test code from testASLmodel (if available)\n');
fprintf('10-18. Classified unlabeled images (see figure)\n');
