%% Verify msseg models given intensity shift perturbation
% https://docs.monai.io/en/1.3.0/transforms.html#shiftintensity
% If understood correctly, just shift every pixel value by the offset.
% x' = x + offset;
%
% Other options to consider:
%
% randshiftintensity
% https://docs.monai.io/en/1.3.0/transforms.html#randshiftintensity
% There are other shift intensity we could consider, but something simple
% like these 2 should be enough
%
% Scale Intensity
% https://docs.monai.io/en/1.3.0/transforms.html#scaleintensity
%
% Random scale intensity
% Pick a random factor (from given bounds) by which to change every pixel
% https://docs.monai.io/en/1.3.0/transforms.html#randscaleintensity
% All pixels in the same image are shifted by same value
%
% Threshold intensity
% https://docs.monai.io/en/1.3.0/transforms.html#thresholdintensity
% This one looks very interesting, similar to brightening/darkening
%
% Scale Intensity Range
% https://docs.monai.io/en/1.3.0/transforms.html#scaleintensityrange
% Simply change the pixel range of the image



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
