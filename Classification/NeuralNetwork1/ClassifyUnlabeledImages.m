%% Classify Unlabeled ASL Images
% This script loads a trained network and classifies the unlabeled images
% with detailed confidence scores to help determine the correct letters.

% Clear workspace and command window
clear;
clc;

% Check if the trained network exists
if ~exist('trainedASLNet.mat', 'file')
    error('Trained network file not found. Please run ASLClassification.m first.');
end

% Load the trained network
load('trainedASLNet.mat', 'trainedNet');

% Define data paths
dataDir = fullfile(pwd, '..', 'data', 'ASL Alphabet', 'Classification');
unlabeledDir = fullfile(dataDir, 'Unlabeled');

% Check if the directory exists
if ~exist(unlabeledDir, 'dir')
    error('Unlabeled directory does not exist. Please check the path.');
end

% Create a datastore for the unlabeled images
unlabeledImds = imageDatastore(unlabeledDir);

% Get the input size required by the network
inputSize = trainedNet.Layers(1).InputSize;

% Resize the unlabeled images
augmentedUnlabeledImds = augmentedImageDatastore(inputSize(1:2), unlabeledImds);

% Classify the unlabeled images
fprintf('Classifying unlabeled images...\n');
[unlabeledPred, unlabeledScores] = classify(trainedNet, augmentedUnlabeledImds);

% Get the class names
classNames = trainedNet.Layers(end).Classes;

% Display the results with detailed confidence scores
numUnlabeledImages = numel(unlabeledImds.Files);
for i = 1:numUnlabeledImages
    fprintf('\nImage %d: %s\n', i, unlabeledImds.Files{i});
    fprintf('Predicted letter: %s (Confidence: %.2f%%)\n', char(unlabeledPred(i)), max(unlabeledScores(i, :)) * 100);

    % Display the top 3 predictions with confidence scores
    [sortedScores, sortedIndices] = sort(unlabeledScores(i, :), 'descend');
    fprintf('Top 3 predictions:\n');
    for j = 1:3
        fprintf('  %s: %.2f%%\n', char(classNames(sortedIndices(j))), sortedScores(j) * 100);
    end
end

% Display the images with their predictions
figure('Name', 'Classified Unlabeled Images', 'Position', [100, 100, 1200, 800]);
for i = 1:numUnlabeledImages
    subplot(3, 3, i);
    img = readimage(unlabeledImds, i);
    imshow(img);

    % Get the top prediction and its confidence
    [sortedScores, sortedIndices] = sort(unlabeledScores(i, :), 'descend');
    topPred = char(classNames(sortedIndices(1)));
    topConf = sortedScores(1) * 100;

    % Get the second prediction and its confidence
    secondPred = char(classNames(sortedIndices(2)));
    secondConf = sortedScores(2) * 100;

    title(sprintf('Image %d: %s (%.1f%%)\n%s (%.1f%%)', i, topPred, topConf, secondPred, secondConf));
end
sgtitle('Classified Unlabeled Images with Top 2 Predictions');

% Run the test model function if available
fprintf('\nRunning testASLmodel function...\n');
try
    testCode = testASLmodel(trainedNet);
    fprintf('Test code from testASLmodel: %s\n', testCode);
catch e
    fprintf('Error running testASLmodel: %s\n', e.message);
end

% Print a summary for the quiz
fprintf('\n--- Quiz Answers for Unlabeled Images ---\n');
fprintf('Enter the following letters in the quiz:\n');
for i = 1:numUnlabeledImages
    fprintf('Image %d: %s\n', i, char(unlabeledPred(i)));
end
