%% Verify msseg models given contrast perturbation
% Main one is Adjust Contrast
% https://docs.monai.io/en/1.3.0/transforms.html#adjustcontrast
% x = ((x - min) / intensity_range) ^ gamma * intensity_range + min
%
% Others to consider
% 
% Mask intensity
% https://docs.monai.io/en/1.3.0/transforms.html#maskintensity
%
% Gibbs Boise
% https://docs.monai.io/en/1.3.0/transforms.html#gibbsnoise
% They mention this is pretty common in 2D/3D MRI data
%
% Foreground Mask
% https://docs.monai.io/en/1.3.0/transforms.html#foregroundmask
% This one seems to change image a lot


rng(0);

% Study variables
N = 10; % number of images to study
sliceSizes = [64, 80, 96]; % for cropping and loading models
imgIdxs = randperm(315,N) + 100; % get 5 images from data
epsilon = [1; 2]; % corresponds to 1 and 2 pixel color values
nPix = {10, 20, 30, 40, 50};

% Define reachability options
% reachOptions = struct;
% reachOptions.reachMethod = 'relax-star-range';
% reachOptions.relaxFactor = 0.95;

%% Reachability analysis for all models

% Begin exploration
for i=1:length(sliceSizes)
    sZ = sliceSizes(i);
    for j = 1:length(epsilon)
        ep = epsilon(j);
        attack = struct;
        attack.epsilon = ep;
        attack.threshold = 55;
        for k = 1:length(nPix)
            nP = nPix{k};
            attack.nPix = nP;
            for m = 1:length(imgIdxs)
                idx = imgIdxs(m);
                reach_model_intensityShift(sZ, idx, reachOptions, attack);
            end
        end
    end
end
