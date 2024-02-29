%% Verify msseg models given a bias field perturbation

rng(0);

% Study variables
N = 2; % number of images to study
sliceSizes = [64, 80, 96]; % for cropping and loading models
imgIdxs = randperm(315,N) + 100; % get N images from data
order = 3; % possible polynomial order values ( > 1, default = 3)
% coeffs = [0, 0.1]; % coefficients for the bias field (typically is within [0,0.1] in projectMonai, but 0.5 for torchIO)
coeff = [0.1, 0.25, 0.5];
coeff_range = [0.00025, 0.0005, 0.001]; % what should the size of this be? 
% 0.001 ~ 2% change for largest interval, ~5 pixel color values
% 0.0005 ~ 1% change, which corresponds to 2.55 pixel color values)
% 0.00025 ~ 1.25 pixel color values
%
% prob = 1; % probability to do random bias field (default is 0.1 in project monai, so only 10% of pixels)
% We assume we modify the whole image

% Perturbation to evaluate
transform = struct;
transform.name = "BiasField";
transform.order = order;
% transform.coefficient = coeff;
% transform.coefficient_range = coeff_range;


%% Reachability analysis for all models

% Define reachability options
reachOptions = struct;
reachOptions.reachMethod = 'relax-star-range';
reachOptions.relaxFactor = 0.95;
% reachOptions.reachMethod = 'approx-star';

% Begin exploration
for i=1:length(sliceSizes)
    
    sZ = sliceSizes(i);

    for j = 1:length(coeff)

        transform.coefficient = coeff(j);

        for k = 1:length(coeff_range)
        
            transform.coefficient_range = coeff_range(k);

            for m = 1:length(imgIdxs)
                
                idx = imgIdxs(m);
                reach_model_monai(sZ, idx, reachOptions, transform);
    
            end

        end

    end
    
end
