%% Count Left-Handed Images in Test Set
% This script helps count the number of left-handed images in the test set
% for letters A and B by displaying them for manual inspection.

% Clear workspace and command window
clear;
clc;

% Define data paths
dataDir = fullfile(pwd, '..', 'data', 'ASL Alphabet', 'Classification');
testDir = fullfile(dataDir, 'Test');

% Check if the directory exists
if ~exist(testDir, 'dir')
    error('Test directory does not exist. Please check the path.');
end

% Create an imageDatastore for letter A in the test set
testADir = fullfile(testDir, 'A');
testAImds = imageDatastore(testADir);

% Create an imageDatastore for letter B in the test set
testBDir = fullfile(testDir, 'B');
testBImds = imageDatastore(testBDir);

% Count the total number of test images for A and B
numTestAImages = numel(testAImds.Files);
numTestBImages = numel(testBImds.Files);

fprintf('Number of test images for letter A: %d\n', numTestAImages);
fprintf('Number of test images for letter B: %d\n', numTestBImages);

% Display images of letter A for manual inspection
fprintf('\nDisplaying images of letter A for manual inspection...\n');
fprintf('Please count the number of left-handed images.\n');

figure('Name', 'Letter A Test Images', 'Position', [100, 100, 1200, 800]);
for i = 1:numTestAImages
    subplot(3, 4, i);
    img = readimage(testAImds, i);
    imshow(img);
    title(sprintf('A Image %d', i));

    % Break if we've displayed 12 images (3x4 grid)
    if i == 12
        break;
    end
end

% Display images of letter B for manual inspection
fprintf('\nDisplaying images of letter B for manual inspection...\n');
fprintf('Please count the number of left-handed images.\n');

figure('Name', 'Letter B Test Images', 'Position', [100, 100, 1200, 800]);
for i = 1:numTestBImages
    subplot(3, 4, i);
    img = readimage(testBImds, i);
    imshow(img);
    title(sprintf('B Image %d', i));

    % Break if we've displayed 12 images (3x4 grid)
    if i == 12
        break;
    end
end

% Instructions for the user
fprintf('\nInstructions:\n');
fprintf('1. Look at each image and determine if it shows a left hand or a right hand.\n');
fprintf('2. Count the number of left-handed images for letter A and letter B.\n');
fprintf('3. Enter these counts in your quiz answers.\n');

% Note: In a left-handed image, the thumb typically appears on the right side of the hand
% when the palm is facing the camera.
