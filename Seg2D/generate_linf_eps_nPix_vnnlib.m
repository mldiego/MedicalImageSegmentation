%% Generate vnnlib specs for semantic segmentation
% Most should be unsat (for single pixel),  most if not all are SAT for all

rng(0);

% Study variables
sliceSizes = [64, 80, 96]; % for cropping and loading models
imgIdxs = randperm(315,5) + 100; % get 5 images from data
epsilon = [0.0001; 0.0005];
nPix = {10, "10", "20"};
% specType = "singlePixel"; % "all", "region"; region = lesion pixels, singlePixel = one pixel from lesion
% specType = "all";
specType = "region";

dataPath = '../../FMitF/Seg2D/data/matData/.._data_axis_2_slice_';% 101.mat';

%% Begin creating vnnlib files for all img (adversarial combos)

% Begin exploration
for i=1:length(sliceSizes)

    sZ = sliceSizes(i); % to choose models and reshape data into

    for j = 1:length(epsilon)

        ep = epsilon(j); % size of attack

        for k = 1:length(nPix)

            nP = nPix{k};
            
            for m = 1:length(imgIdxs)

                idx = imgIdxs(m);

                create_pred_vnnlib(dataPath, sZ, idx, ep, nP, specType);

            end

        end

    end

end

%% Helper functions

function create_pred_vnnlib(dataPath, sliceSize, imgIdx, epsilon, nPix, specType)

    rng(2024); % to select the same pixels for all models

    % get number of pixels
    if isa(nPix, 'string') % strings define percentage of pixels in image
        nPix = floor(sliceSize^2 * str2double(nPix)/100);
    end

    % create file name
    name = "vnnlib/img_"+string(imgIdx)+"_sliceSize_"+string(sliceSize) + ...
        "_linf_pixels_" + string(nPix)+ "_eps_" + string(epsilon) + "_" + specType;
    vnnlibfile = name+".vnnlib";

    % Load data
    data = load([dataPath num2str(imgIdx) '.mat']);
    img = data.flair; % input
    target = data.mask; % output (target)
    img_size = size(img); % size of I/O
    
    % Preprocess data
    target_size = [sliceSize sliceSize];
    r = centerCropWindow2d(img_size, target_size); % cropping window
    slice_img = imcrop(img, r); % sliced input image
    slice_target = imcrop(target, r); % sliced target output
    idxs = randperm(numel(slice_img), nPix); % idxs to modify in input image

    % create vnnlib files
    sliceIn = reshape(slice_img, [], 1);
    sliceOut = reshape(slice_target, [], 1);
    outputSpec = create_output_spec(sliceOut, specType);
    if isempty(outputSpec)
        warning("No interesting region to return for "+vnnlibfile);
        return
    end
    [lb,ub] = l_inf_attack(sliceIn, idxs, epsilon, inf, 0); % no known upper bound

    % Save spec
    export2vnnlib(lb, ub, length(ub), outputSpec, vnnlibfile);
    disp("Created property "+vnnlibfile);

end



%% Helper functions

% Return the bounds an linf attack
function [lb,ub] = l_inf_attack(img, idxs, epsilon, max_value, min_value)
    imgSize = size(img);
    pix2attack = zeros(imgSize, "like", img);
    pix2attack(idxs) = 1;
    disturbance = epsilon * pix2attack; % disturbance value
    lb = max(img - disturbance, min_value);
    ub = min(img + disturbance, max_value);
end

% Define property 
function Hs = create_output_spec(mask, specType)
    % @Hs: unsafe/not robust region defined as a HalfSpace
    % This should be 1 halfspace for each pixel?

    outSize = length(mask);
    xx = find(mask > 0); % important class (used for pixel and region specType)

    if isempty(xx)
        Hs = []; % return empty output spec
        return
    end

    if strcmp(specType, "singlePixel")

        % initalize halfspace vars
        g = 0; % threshold to determine class
        G = zeros(1,outSize);
        % define spec
        x = xx(1); % only choose first pixel to verify
        G(x) = 1;
        Hs = HalfSpace(G,g);

    elseif strcmp(specType, "region")
        Hs = [];
        % initalize halfspace vars
        G = zeros(length(xx), outSize);
        g = zeros(length(xx),1);
        Hs = [];
        % define spec
        for i=1:length(xx)
            x = xx(i); 
            G(i, x) = 1;
            Hs = [Hs; HalfSpace(G(i,:), g(i))];
        end

    elseif strcmp(specType, "all")
        Hs = [];
        % initalize halfspace vars
        G = zeros(outSize, outSize);
        g = zeros(outSize,1);
        % define spec
        for i=1:outSize
            if sum(ismember(xx, i)) > 0
                G(i, i) = 1;
            else
                G(i, i) = -1;
            end
            Hs = [Hs; HalfSpace(G(i,:), g(i))];
        end

    end

end

