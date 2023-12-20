%% Load image data 
% both image and mask
dataPath = '../../FMitF/Seg2D/data/matData';
imgName = '.._data_axis_2_slice_101.mat';
data = load(fullfile(dataPath, imgName));
img = data.flair;
target = data.mask;
% img_ = reshape(img, [], 1);
% target_ = reshape(target, [], 1);
img_size = size(img);

dims = [64, 80, 96, 112, 128];
epsilon = 0.001;
% nPix = [10, 20, 30];

% only one datapoint for now
for k=1:length(dims)
    
    % create file name
    name = "vnnlib/slice101_" + dims(k) + "_linf_" + string(epsilon);
    vnnlibfile = name+".vnnlib";
    
    % Preprocess data
    target_size = [dims(k) dims(k)];
    r = centerCropWindow2d(img_size, target_size);
    slice_img = imcrop(img, r);
    slice_target = imcrop(target, r);
    
    % create vnnlib files
    sliceIn = reshape(slice_img, [], 1);
    sliceOut = reshape(slice_target, [], 1);
    outputSpec = create_output_spec(sliceOut);
    [lb,ub] = l_inf_attack(sliceIn, epsilon, 1, 0);
    
    % Save spec
    export2vnnlib(lb, ub, length(ub), outputSpec, vnnlibfile);
    disp("Created property "+vnnlibfile);

end




%% Helper functions

% Return the bounds an linf attack
function [lb,ub] = l_inf_attack(img, epsilon, max_value, min_value)
    imgSize = size(img);
    disturbance = epsilon * ones(imgSize, "like", img); % disturbance value
    lb = max(img - disturbance, min_value);
    ub = min(img + disturbance, max_value);
end

% Define unsafe (not robust) property 
function Hs = create_output_spec(mask)
    % @Hs: unsafe/not robust region defined as a HalfSpace
    % This should be 1 halfspace for each pixel?

    outSize = length(mask);
    xx = find(mask >= 1); % important class

    g = 0; % threshold to determine class
    G = zeros(1,outSize);

    if isempty(xx)
        xx = 1; % first pixel
        G(xx) = -1;
    else
        xx = xx(1); % only choose first pixel to verify
        G(xx) = 1;
    end

    % Create HalfSapce to define robustness specification
    Hs = HalfSpace(G, g);

end