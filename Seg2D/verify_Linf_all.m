%% Questions
% 1) How are the slices cropped? 
% Original slices are 224x128, but input to NN is 128x128
%  - It is like a moving window all accross the image
% 
% We can rotate imrotate to rotate images, can specify method and angle to rotate each image with
% 2) What id the post processing done after inference on the model?
%  - Do rotation for image registration
%  - Focus on adversarial attacks or different brightening processes for
%  creating the input sets for segmentation

rng(0);

% Study variables
sliceSizes = [64, 80, 96, 112, 128]; % for cropping and loading models
imgIdxs = randperm(315,5) + 100; % get 5 images from data
epsilon = [0; 0.0001; 0.0005];
nPix = {10, 20, 40, 80, "10", "20", "50", "100"}; % strings define percentage of pixels in image

% Define reachability options
reachOptions = struct;
reachOptions.reachMethod = 'relax-star-range';
relaxFactors = 1;

%% Reachability analysis for all models

% Begin exploration
for i=1:length(sliceSizes)
    sZ = sliceSizes(i);
    for j = 1:length(epsilon)
        eps = epsilon(j);
        attack = struct;
        attack.epsilon = eps;
        for k = 1:length(nPix)
            nP = nPix{k};
            attack.nPix = nP;
            for m = 1:length(imgIdxs)
                idx = imgIdxs(m);
                reach_model_instance(sZ, idx, reachOptions, attack);
            end
        end
    end
end
