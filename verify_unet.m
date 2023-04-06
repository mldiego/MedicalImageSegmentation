%% Robustness verification of unet 
% Semantic segmentation task to predict the area of the triangles 
% triangle dataset --> fullfile(toolboxdir('vision'),'visiondata','triangleImages');

% 1) Load model
model = load('unet_avg.mat');
net = matlab2nnv(model.net);

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
XData = cell(n,1);
YData = cell(n,1);
for i=1:n
    XData{i} = imread(imds.Files{idxs(i)});
    YData{i} = imread(pxds.Files{idxs(i)});
end
    
% 3) Create input set
npixels = 10; % number of pixels to attack
pix_idxs = randperm(1024,npixels); % randomly select pixels to attack
disturbance = 0.0001;
lb_ = zeros(32,32);
ub_ = zeros(32,32);
I(n) = ImageStar;
for i=1:n
    im = double(XData{i});
    lb = lb_;
    lb(pix_idxs) = -disturbance;
    ub = ub_;
    ub(pix_idxs) = disturbance;
    I(i) = ImageStar(im, lb, ub);
end

% 4) Verify network
time = zeros(n,1);        % computation time
riou = zeros(n,1);        % intersection over union
rv = zeros(n,1);          % robustness value
rs = zeros(n,1);          % sensitivity
ver_rs = cell(n,1);       % verified reach set
eval_seg_ims = cell(n,1); % predicted segmentation image
f1 = cell(n,1);           % figure 1 (evaluated image)
f2 = cell(n,1);           % figure 2 (verified image w.r.t evaluated image)
reachOptions.reachMethod = 'approx-star';
for i=1:n
    t = tic;
    [riou(i), rv(i), rs(i), ~, ~, ~, ~, ver_rs{i}, eval_seg_ims{i}] = net.verify_segmentation(I(i), XData(i), reachOptions);
    time(i) = toc(t);
     % Visualize results
    [f1{i}, f2{i}] = net.plot_segmentation_output_set(ver_rs{1}, eval_seg_ims{1});
end

% Save results
save("unet_robustness_verify.mat", 'time', 'riou', 'rs', 'rv', 'f1', 'f2');

