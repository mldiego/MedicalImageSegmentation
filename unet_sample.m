%% Example 1 - Create U-Net with custom encoder/decoder depth.

    % Create U-Net layers with an encoder/decoder depth of 3.
    inputSize = [480 640 3];
    numClasses = 5;
    encoderDepth = 3;
    lgraph = unetLayers(inputSize, numClasses, 'EncoderDepth',...
    encoderDepth);
 
    % Display network.
    analyzeNetwork(lgraph)
 
%% Example 2 - Create U-Net with "valid" convolution layers.
    
    % Create U-Net layers with an encoder/decoder depth of 4.
    inputSize = [572 572 3];
    numClasses = 5;
    encoderDepth = 4;
    lgraph2 = unetLayers(inputSize, numClasses, 'EncoderDepth', ...
    encoderDepth, 'ConvolutionPadding','valid');
 
    % Display network.
    analyzeNetwork(lgraph2)
 
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
    net = trainNetwork(ds, lgraph, options);