%% Example - Train U-Net.
    
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
% Replace maxpool layers with avgpool
avgP1 = averagePooling2dLayer(2, 'Name', 'avgP1', 'Stride', 2);
lgraph = replaceLayer(lgraph, 'Encoder-Stage-1-MaxPool', avgP1);
avgP2 = averagePooling2dLayer(2, 'Name', 'avgP2', 'Stride', 2);
lgraph = replaceLayer(lgraph, 'Encoder-Stage-2-MaxPool', avgP2);
avgP3 = averagePooling2dLayer(2, 'Name', 'avgP3', 'Stride', 2);
lgraph = replaceLayer(lgraph, 'Encoder-Stage-3-MaxPool', avgP3);
avgP4 = averagePooling2dLayer(2, 'Name', 'avgP4', 'Stride', 2);
lgraph = replaceLayer(lgraph, 'Encoder-Stage-4-MaxPool', avgP4);

% Combine image and pixel label data to train a semantic segmentation
% network.
ds = combine(imds,pxds);

% Setup training options.
options = trainingOptions('sgdm', 'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 20, 'VerboseFrequency', 10);
rng(0);
% Train network.
[net, info] = trainNetwork(ds, lgraph, options);
accuracy = info.TrainingAccuracy(end);
% model returned is the last one, so save that training accuracy

% Save network
save('unet_avg.mat','net', 'accuracy');
