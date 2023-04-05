%% Example 3 - Train U-Net.
    
% Load training images and pixel labels.
dataSetDir = fullfile(toolboxdir('vision'),'visiondata','triangleImages');
imageDir = fullfile(dataSetDir, 'trainingImages');
labelDir = fullfile(dataSetDir, 'trainingLabels');

% Create an imageDatastore holding the training images.
imds = imageDatastore(imageDir);

% Define the class names and their associated label IDs.
classNames = ["triangle", "background"];
labelIDs   = [255 0];

% Create a pixelLabelDatastore holding the ground truth pixel labels
% for the training images.
pxds = pixelLabelDatastore(labelDir, classNames, labelIDs);

% Create U-Net.
inputSize = [32 32];
numClasses = 2;
[lgraph,~] = unetLayers(inputSize, numClasses);

% Combine image and pixel label data to train a semantic segmentation
% network.
ds = combine(imds,pxds);

% Setup training options.
options = trainingOptions('sgdm', 'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 20, 'VerboseFrequency', 10);

% Train network.
[net, info] = trainNetwork(ds, lgraph, options);
accuracy = info.TrainingAccuracy(end);
% model returned is the last one, so save that training accuracy

% Save network
save('unet.mat','net', 'accuracy');
