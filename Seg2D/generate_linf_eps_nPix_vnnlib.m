%% Generate vnnlib specs for demantic segmentation
% Most should be unsat

rng(0);

% Study variables
sliceSizes = [64, 80, 96]; % for cropping and loading models
imgIdxs = randperm(315,5) + 100; % get 5 images from data
epsilon = [0.0001; 0.0005];
nPix = {10, "10", "20"};

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

                create_vnnlib(dataPath, sZ, idx, ep, nP);

            end
        end
    end
end

%% Helper functions

function create_vnnlib(dataPath, sliceSize, imgIdx, epsilon, nPix)

    rng(2024); % to select the same pixels for all models

    % get number of pixels
    if isa(nPix, 'string') % strings define percentage of pixels in image
        nPix = floor(sliceSize^2 * str2double(nPix)/100);
    end

    % create file name
    name = "vnnlib/img_"+string(imgIdx)+"_sliceSize_"+string(sliceSize) + "_linf_pixels_" + string(nPix)+ "_eps_" + string(epsilon);
    vnnlibfile = name+".vnnlib";

    % Load data
    data = load([dataPath num2str(imgIdx) '.mat']);
    img = data.flair; % input
    target = data.mask; % output (target)
    img_size = size(img); % size of I/O
    
    % Preprocess data
    target_size = [dims(k) dims(k)];
    r = centerCropWindow2d(img_size, target_size); % cropping window
    slice_img = imcrop(img, r); % sliced input image
    slice_target = imcrop(target, r); % sliced target output
    idxs = randperm(numel(slice_img), nPix); % idxs to modify in input image

    % create vnnlib files
    sliceIn = reshape(slice_img, [], 1);
    sliceOut = reshape(slice_target, [], 1);
    outputSpec = create_output_spec(sliceOut);
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

