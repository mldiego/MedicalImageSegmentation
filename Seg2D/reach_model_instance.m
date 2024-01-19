function [R,rT] = reach_model_instance(sliceSize, imgIdx, reachOptions, attack)
   
    %% Load network
    
    netMatlab = importONNXNetwork("models/size_"+string(sliceSize)+"/best_model.onnx");
    net = matlab2nnv(netMatlab);
    
    %% Load data
    
    % load single data point
    data_path = "../../FMitF/Seg2D/data/matData/";
    img_path = data_path + ".._data_axis_2_slice_"+string(imgIdx)+".mat";
    data = load(img_path);
    img = data.flair;
    
    img_size = size(img);
    target_size = [sliceSize, sliceSize];
    r = centerCropWindow2d(img_size, target_size);
    slice_img = imcrop(img, r);

    % target = img.mask;
    % slice_target = imcrop(target, r);
    % y = netMatlab.predict(slice_img);
    % y_mask = double(y > 0);

    % Visualize images
    % figure;
    % subplot(2,2,1)
    % imshow(slice_img, [min(slice_img, [], 'all') max(slice_img, [], 'all')])
    % subplot(2,2,2)
    % imshow(slice_target, [min(slice_target, [], 'all') max(slice_target, [], 'all')])
    % subplot(2,2,3)
    % imshow(y, [min(y, [], 'all') max(y, [], 'all')])
    % subplot(2,2,4)
    % imshow(y_mask, [min(y_mask, [], 'all') max(y_mask, [], 'all')])

    %% Get attack data
    
    nPix = attack.nPix;
    epsilon = attack.epsilon;
    % todo: add other types of attacks
    
    % Select random pixels to attack
    rng(2024); % to select the same pixels for all models
    if isa(nPix, 'string') % strings define percentage of pixels in image
        nPix = floor(sliceSize^2 * str2double(nPix)/100);
    end

    idxs = randperm(numel(slice_img), nPix);
        
    %% Define input set
    
    % Create bounds
    img_lb = slice_img;
    img_lb(idxs) = img_lb(idxs) - epsilon;
    img_ub = slice_img;
    img_ub(idxs) = img_ub(idxs) + epsilon;
    
    % Create input set
    IS = ImageStar(img_lb, img_ub);

    %% Compute reach sets
    
    t = tic;
    ME = [];
    try
        R = net.reach(IS, reachOptions);
    catch ME
        R = net.reachSet;
    end
    rT = toc(t);

    save("results/reach_linf_"+string(sliceSize)+"_"+string(imgIdx)+"_"+string(nPix)+"_"+string(epsilon)+".mat", "R", "rT", "ME", "-v7.3");


end

