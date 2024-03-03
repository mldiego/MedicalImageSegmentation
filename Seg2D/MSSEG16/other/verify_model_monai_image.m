function verify_model_monai_image(sliceSize, imgIdx, reachMethod, relaxFactor, transformType, varargin)
   
    %% Load network
    
    netMatlab = importNetworkFromONNX("models/size_"+string(sliceSize)+"/best_model.onnx");
    net = matlab2nnv(netMatlab);
    
    %% Load data
    
    % load single data point
    data_path = "../../FMitF/Seg2D/data/matData/";
    img_path = data_path + ".._data_axis_2_slice_"+string(imgIdx)+".mat";
    data = load(img_path);
    img = data.flair;
    
    img_size = size(img);
    target_size = [str2double(sliceSize), str2double(sliceSize)];
    r = centerCropWindow2d(img_size, target_size); % This fails when ran from the command line...
    slice_img = imcrop(img, r);
        
    %% Define input set

    %%%%%%%%%%%%%
    %%% TODO %%%%
    %%%%%%%%%%%%%

    switch transformType
        case "IntensityShift"
            % IS = bright_attack(slice_img, nPix, threshold, epsilon);
            error("Working on it");
        case "AdjustContrast"
            gamma = varargin{1};
            gamma_range = varargin{2};
            IS = AdjustContrast(slice_img, gamma, gamma_range);
        case "BiasField"
            order = varargin{1};
            coefficient = varargin{2};
            coefficient_range = varargin{3};
            IS = BiasField(slice_img, order, coefficient, coefficient_range);
        otherwise
            error("Wrong transformation name. Only 'AdjustContrast' , 'BiasField', and 'IntensityShift' are supported yet.");
    end
    

    %% Compute reach sets
    
    % get reachability options
    reachOptions = struct;
    reachOptions.reachMethod = reachMethod;
    reachOptions.relaxFactor = str2double(relaxFactor);
    
    t = tic;
    ME = [];
    try
        R = net.reach(IS, reachOptions);
    catch ME
        % R = net.reachSet;
        R = [];
    end
    rT = toc(t);

    if strcmp(transformType, "BiasField")
        save("results/reach_monai_"+transformType+"_"+sliceSize+"_" + imgIdx + "_"+order+"_" + coefficient+"_"...
        +coefficient_range + "_" + reachMethod + relaxFactor+".mat", "R", "rT", "ME", "-v7.3");
    elseif strcmp(transformType, "AdjustContrast")
        save("results/reach_monai_"+transformType+"_"+sliceSize+"_" + imgIdx + "_"+order+"_" + coefficient+"_"...
        +coefficient_range + "_" + reachMethod + relaxFactor+".mat", "R", "rT", "ME", "-v7.3");
    elseif strcmp(transformType, "linf")
        save("results/reach_monai_"+transformType+"_"+sliceSize+"_" + imgIdx + "_"+order+"_" + coefficient+"_"...
        +coefficient_range + "_" + reachMethod + relaxFactor+".mat", "R", "rT", "ME", "-v7.3");
    end


end

%% Transform functions

% function I = IntensityShift()
% end

function I = BiasField(img, order, coeffs, cRange)
% Get the code from torchIO to generate the BiasField
% https://torchio.readthedocs.io/_modules/torchio/transforms/augmentation/intensity/random_bias_field.html#RandomBiasField

    % Transform variables
    coeffs = str2double(coeffs);
    order  = str2double(order);
    cRange = str2double(cRange);
    
    bField1 = generate_bias_field(img, coeffs-cRange, order);
    img1 = img .* bField1;

    bField2 = generate_bias_field(img, coeffs+cRange, order);
    img2 = img .* bField2;

    % get max and min value for every pixel given the biasField applied
    % interval range for every pixel is given by min and max values for
    % that pixel in images img1 and img2
    img_lb = min(img1,img2);
    img_ub = max(img1,img2);

    % Define input set as ImageStar
    img_diff = img_ub - img_lb;
    V(:,:,:,1) = img_lb; % assume lb is center of set (instead of img)
    V(:,:,:,2) = img_diff ; % basis vectors
    C = [1; -1]; % constraints
    d = [1; -1];
    I = ImageStar(V, C, d, 0, 1); % input set

end

function bias_field = generate_bias_field(img, coeffs, order)
% def generate_bias_field(
%     data: TypeData,
%     order: int,
%     coefficients: TypeData,
% ) -> np.ndarray:
%     # Create the bias field map using a linear combination of polynomial
%     # functions and the coefficients previously sampled
%     shape = np.array(data.shape[1:])  # first axis is channels
%     half_shape = shape / 2
% 
%     ranges = [np.arange(-n, n) + 0.5 for n in half_shape]
% 
%     bias_field = np.zeros(shape)
%     meshes = np.asarray(np.meshgrid(*ranges))
% 
%     for mesh in meshes:
%         mesh_max = mesh.max()
%         if mesh_max > 0:
%             mesh /= mesh_max
%     x_mesh, y_mesh, z_mesh = meshes
% 
%     i = 0
%     for x_order in range(order + 1):
%         for y_order in range(order + 1 - x_order):
%             for z_order in range(order + 1 - (x_order + y_order)):
%                 coefficient = coefficients[i]
%                 new_map = (
%                     coefficient
%                     * x_mesh**x_order
%                     * y_mesh**y_order
%                     * z_mesh**z_order
%                 )
%                 bias_field += np.transpose(new_map, (1, 0, 2))  # why?
%                 i += 1
%     bias_field = np.exp(bias_field).astype(np.float32)
%     return bias_field

    % Create the bias field map using a linear combination of polynomial
    % functions and the coefficients previously sampled
    shape = size(img); 
    % shape = shape(2:end); % first axis is channels (not necessary as it is a greyscale image -> 1 channel, already removed dimension)
    half_shape = shape ./ 2;
    ranges = [];
    
    for n = half_shape
        ranges = [ranges; (-n:1:(n-1)) + 0.5];
    end
    
    bias_field = zeros(shape);
    
    ndim = length(shape);
    meshes = zeros([ndim, shape(1), shape(2)]);
    for k=1:ndim
        meshes(k, :,:) = meshgrid(ranges(k,:));
    end
    
    for i = 1:size(meshes,1)
        mesh = meshes(i,:,:);
        mesh_max = max(mesh, [], 'all');
        if mesh_max > 0
            mesh = mesh./mesh_max;
            meshes(i,:,:) = mesh;
        end
    end
    
    x_mesh = meshes(1,:,:);
    y_mesh = meshes(2,:,:);
    
    % i = 0; % initialize counter (for coeffs)
    cf = coeffs;
    for x_order = 0:order
        for y_order = 0:(order-x_order)
            % cf = coeffs(i); % will this always be within limits? Why this?
            new_map = (cf .* x_mesh.^x_order .* y_mesh.^y_order);
            new_map = squeeze(permute(new_map, [2 1 3]));
            bias_field = bias_field + new_map; % why?
            % i = i + 1;
        end
    end
    
    % And that's it
    bias_field = exp(bias_field);

end

function I = AdjustContrast(img, gamma, gamma_range)
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
    gamma = str2double(gamma);
    gamma_range = str2double(gamma_range);
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


