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
    
    netMatlab = importNetworkFromONNX("models/model"+string(sliceSize)+".onnx");
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
    % IS = ImageStar(lb, ub); % Out of memory for some inputs
    % Dimensions of imagestar
    h = size(lb,1);
    w = size(lb,2);
    c = 1;
    
    % create ImageStar variables
    center = cast(0.5*(ub+lb), 'like', lb);
    v = 0.5*(ub-lb);
    idxRegion = find((lb-ub)); % meaning: no input set, simply inference, so skip for now, this will be computed just for the metrics
    
    if ~isempty (idxRegion)
        n = length(idxRegion);
        V = zeros(h, w, c, n, 'like', center);
        bCount = 1;
        for i1 = 1:size(v,1)
            for i2 = 1:size(v,2)
                basisValue = v(i1,i2);
                if basisValue
                    V(i1,i2,:,bCount) = v(i1,i2);
                    bCount = bCount + 1;
                end
            end
        end
        C = zeros(1, n, 'like', V);
        d = zeros(1, 1, 'like', V);
        % Construct ImageStar
        IS = ImageStar(cat(4,center,V), C, d, -1*ones(n,1), ones(n,1), lb, ub);
    
        %% Compute reach sets
        
        % get reachability options
        reachOptions = struct;
        reachOptions.reachMethod = reachMethod;
        reachOptions.relaxFactor = str2double(relaxFactor);
        reachOptions.lp_solver = 'gurobi';
        
        t = tic;
        ME = [];
        try
            R = net.reach(IS, reachOptions);
            [lb,ub] = R.estimateRanges;
        catch ME
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

    end % skip any reachability analysis if lower bound = upper bound 

end

