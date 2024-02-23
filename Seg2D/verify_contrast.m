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


rng(0);

% Study variables
N = 10; % number of images to study
sliceSizes = [64, 80, 96]; % for cropping and loading models
imgIdxs = randperm(315,N) + 100; % get N images from data
gamma = [0.5; 1; 2]; % lower and upper bound for typical ranges used for gamma
gamma_range = [0.0075; 0.01; 0.0125; 0.015]; % gamma ranges to consider for each gamma value
% first one is equivalent to max interval ranges of about 1.3935 (for digital images)


% Define reachability options
reachOptions = struct;
reachOptions.reachMethod = 'relax-star-range';
reachOptions.relaxFactor = 0.8;
% reachOptions.reachMethod = 'approx-star';

%% Reachability analysis for all models

% Perturnation to evaluate
transform = struct;
transform.name = "AdjustContrast";

% Begin exploration
for i=1:length(sliceSizes)
    
    sZ = sliceSizes(i);

    for j = 1:length(gamma)
        
        transform.gamma = gamma(j);

        for k = 1:length(gamma_range)

            transform.gamma_range = gamma_range(k);

            for m = 1:length(imgIdxs)
                
                idx = imgIdxs(m);
                reach_model_monai(sZ, idx, reachOptions, transform);
            
            end

        end

    end
    
end
