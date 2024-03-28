function verify_model_subject_patch(img_path, subject, sliceSize, reachMethod, relaxFactor, transformType, varargin)
 
    % We are changing our evaluation to verify whole subject data using sliding window

    % Get variable arguments
    switch transformType
        case "linf"
            epsilon = varargin{1};
            nPix = varargin{2}; % percentage for each 2D data (not slices)
        case "AdjustContrast"
            gamma = varargin{1};
            gamma_range = varargin{2};
        case "BiasField"
            order = varargin{1};
            coefficient = varargin{2};
            coefficient_range = varargin{3};
        otherwise
            error("Wrong transformation name. Only 'AdjustContrast' , 'BiasField', and 'IntensityShift' are supported yet.");
    end
 
    %% Load network
    
    netMatlab = importNetworkFromONNX("models/size_"+string(sliceSize)+"/best_model.onnx");
    net = matlab2nnv(netMatlab);
    
    %% Load data
    
    % load single patch
    data = load(img_path);
    lb = data.lb;
    ub = data.ub;
    
    % get info from data
    info = split(img_path, "_");
    channel = info{2};
    xC = info{3};
    yC = info{4};
    yC = split(yC, '.');
    yC = yC{1};


    %% Define input set as ImageStar
    img_diff = ub - lb;
    V(:,:,:,1) = lb; % assume lb is center of set (instead of img)
    V(:,:,:,2) = img_diff ; % basis vectors
    C = [1; -1]; % constraints
    d = [1; -1];
    IS = ImageStar(V, C, d, 0, 1); % input set
    

    %% Compute reach sets
    
    % get reachability options
    reachOptions = struct;
    reachOptions.reachMethod = reachMethod;
    reachOptions.relaxFactor = str2double(relaxFactor);
    
    t = tic;
    ME = [];
    try
        R = net.reach(IS, reachOptions);
        [lb, ub] = R.estimateRanges;
    catch ME
        % R = net.reachSet;
        lb = []; ub = [];
    end
    rT = toc(t);


    if strcmp(transformType, "BiasField")
        save("results/reach_monai_" + transformType + "_" + sliceSize+ "_" + subject + "_" ...
        + channel + "_" + xC + "_" + yC + "_" + order + "_" + coefficient + "_" + coefficient_range...
        + "_" + reachMethod + relaxFactor+".mat", "lb", "ub", "rT", "ME", "-v7.3");
    elseif strcmp(transformType, "AdjustContrast")
        save("results/reach_monai_" + transformType + "_" + sliceSize+ "_" + subject + "_" ...
        + channel + "_" + xC + "_" + yC + "_" + gamma + "_" + gamma_range...
        + "_" + reachMethod + relaxFactor+".mat", "lb", "ub", "rT", "ME", "-v7.3");
    elseif strcmp(transformType, "linf")
        save("results/reach_monai_" + transformType + "_" + sliceSize+ "_" + subject + "_" ...
        + channel + "_" + xC + "_" + yC + "_" + epsilon + "_" + nPix...
        + "_" + reachMethod + relaxFactor+".mat", "lb", "ub", "rT", "ME", "-v7.3");
    end


end

