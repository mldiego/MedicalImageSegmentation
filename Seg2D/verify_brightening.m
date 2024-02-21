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
N = 10; % number of images to study
sliceSizes = [64, 80, 96]; % for cropping and loading models
imgIdxs = randperm(315,N) + 100; % get 5 images from data
epsilon = [1/(255)^2; 2/(255)^2]; % because data is normalized, this actually corresponds to 1 and 2 pixel color values
nPix = {10, 20, 30, 40, 50};

% Define reachability options
reachOptions = struct;
reachOptions.reachMethod = 'relax-star-range';
reachOptions.relaxFactor = 0.95;

%% Reachability analysis for all models

% Begin exploration
for i=1:length(sliceSizes)
    sZ = sliceSizes(i);
    for j = 1:length(epsilon)
        ep = epsilon(j);
        attack = struct;
        attack.epsilon = ep;
        attack.threshold = 100;
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
