function [R,rT] = reach_model_monai(sliceSize, imgIdx, reachOptions, transform)
   
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
        
    %% Define input set

    %%%%%%%%%%%%%
    %%% TODO %%%%
    %%%%%%%%%%%%%

    switch transform.name
        case "IntensityShift"
            % IS = bright_attack(slice_img, nPix, threshold, epsilon);
            error("Working on it");
        case "AdjustContrast"
            IS = AdjustContrast(slice_img, transform);
        otherwise
            error("Wrong transformation name. Only 'AdjustContrast' and 'IntensityShift' are supported yet.");
    end
    

    %% Compute reach sets
    
    t = tic;
    ME = [];
    try
        R = net.reach(IS, reachOptions);
    catch ME
        R = net.reachSet;
    end
    rT = toc(t);

    save("results/reach_monai_"+transform.name+"_"+string(sliceSize)+"_"+...
        string(imgIdx)+"_"+string(nPix)+"_"+string(epsilon)+".mat", "R", "rT", "ME", "-v7.3");


end

%% Transform functions

% function I = IntensityShift()
% end

function I = AdjustContrast(img, transform)
% Python Code from Project-MONAI
    % """
    % Apply the transform to `img`.
    % gamma: gamma value to adjust the contrast as function.
    % """
    % img = convert_to_tensor(img, track_meta=get_track_meta())
    % gamma = gamma if gamma is not None else self.gamma
    % 
    % if self.invert_image:
    %     img = -img
    % 
    % if self.retain_stats:
    %     mn = img.mean()
    %     sd = img.std()
    % 
    % epsilon = 1e-7
    % img_min = img.min()
    % img_range = img.max() - img_min
    % ret: NdarrayOrTensor = ((img - img_min) / float(img_range + epsilon)) ** gamma * img_range + img_min
    % 
    % if self.retain_stats:
    %     # zero mean and normalize
    %     ret = ret - ret.mean()
    %     ret = ret / (ret.std() + 1e-8)
    %     # restore old mean and standard deviation
    %     ret = sd * ret + mn
    % 
    % if self.invert_image:
    %     ret = -ret
    % 
    % return ret

    % Our Implementation
    img_min = min(img, [], 'all');
    img_max = max(img, [], 'all');
    img_range = img_max-img_min;
    epsilon = 1e-7; % not sure why we need this, but to avoid dividing by zero probably ???
    gamma = transform.gamma;
    gamma_range = transform.gamma_range;
    if gamma == 2 % upper range
        gamma = gamma - gamma_range;
    end
    % The range for gamma is 0.5 to 2 as default for most code I have seen
    
    % This is the transformed image. 
    % img_trans = ((img-img_min)./(img_range+epsilon)).^gamma * img_range + img_min;

    % Do we create the set from here by assuiming gama is a set of values?
    % This may make the most sense...
    % I believe this would assign no range values for the background
    % 
    % Would this look something like...
    img1 = ((img-img_min)./(img_range+epsilon)).^gamma * img_range + img_min;
    img2 = ((img-img_min)./(img_range+epsilon)).^(gamma+gamma_range) * img_range + img_min;
    % IS = ImageStar(lb,ub);
    
    % However, this will not be computationally efficient (defintely not as
    % efficient as the bright/dark perturbations)
    
    % Would this work?

    % Define input set as ImageStar
    img_diff = img1 - img2;
    V(:,:,:,1) = img2; % assume lb is center of set (instead of img)
    V(:,:,:,2) = img_diff ; % basis vectors
    C = [1; -1]; % constraints
    d = [1; -1];
    I = ImageStar(V, C, d, 0, 1); % input set

    % Is this okay? Check point
    % [lb1, ub1] = I.estimateRanges;
    % lb_diff = lb1 - img2;
    % ub_diff = ub1 - img1;

end


