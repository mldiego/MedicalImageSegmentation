%% Robustness verification of unet 
% Semantic segmentation task to predict the area of the triangles 
% triangle dataset --> fullfile(toolboxdir('vision'),'visiondata','triangleImages');

% 1) Load model
% model = load('unet.mat');
% net = matlab2nnv(model.net);

% 2) Load data
dataSetDir = fullfile(toolboxdir('vision'),'visiondata','triangleImages');
imageDir = fullfile(dataSetDir, 'trainingImages');
labelDir = fullfile(dataSetDir, 'trainingLabels');
imds = imageDatastore(imageDir);
% Define the class names and their associated label IDs.
classNames = ["triangle", "background"];
labelIDs   = [255 0];
pxds = pixelLabelDatastore(labelDir, classNames, labelIDs);
ds = combine(imds,pxds);
% This just holds the path to files, load a few images from there and
% verify them uner some attack/noise

% 2a) Load images to evaluate
N = 200; % number of images in dataset
n = 5; % number of images to evaluate
rng(0);
idxs = randperm(N,n);
XData = zeros(32, 32, n);
YData = zeros(32, 32, n);
for i=idxs
    XData(:,:,i) = imread(imds.Files{i});
    YData(:,:,i) = imread(pxds.Files{i});
end
    
% 3) Create input set
npixels = 10; % number of pixels to attack
pix_idxs = randperm(1024,npixels); % randomly select pixels to attack

