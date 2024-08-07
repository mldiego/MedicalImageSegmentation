%% Get transform stats for msseg models given a bias field perturbation

% Study variables
% order = "3"; % possible polynomial order values ( > 1, default = 3)
% var1 = [0.1, 0.25, 0.5]; % coeff
% var2 = [0.00025, 0.0005, 0.001]; % coeff_range
% transType = "BiasField";

% var1 = [0.002; 0.004; 0.006]; % (epsilon) equivalent to 1,2, 3 pixel color values
% var2 = [5, 10, 15]; (nPix)
% transType = "linf";

var1 = [0.5; 1; 2]; % (gamma)
% var2 = [0.0025; 0.00375; 0.005]; % gamma ranges to consider for each gamma value
var2 = 0.0025;
transType = "AdjustContrast";

% Data
path2data = "../../data/ISBI/subjects/01/";
subjects = ["01", "02", "03", "04"]; % subject data to analyze (omly use mask1 for each)


for s = 1:length(subjects)

    sb = subjects(s);
    sbName = split(sb, '/');
    sbName = string(sbName{1});

    % load 3d data
    flair = niftiread(path2data + sb+"/flair.nii");
    flair = flair_normalization(flair);
        
    for j = 1:length(var1)

        v1 = var1(j);
        v1 = string(v1);

        for k = 1:length(var2)
        
            v2 = var2(k);
            v2 = string(v2);

            stats = zeros(size(flair,1), 6); % stats [max_diff, mean_diff, median_diff, max_flair]

            % t = tic;

            for c = 1:size(flair,1) % iterate through all 2D slices 
                if strcmp(transType, "BiasField")
                    [lb, ub] = BiasField(flair(c,:,:), order, v1, v2);
                elseif strcmp(transType, "AdjustContrast")
                    [lb, ub] = AdjustContrast(flair(c,:,:), v1, v2);
                elseif strcmp(transType, "linf")
                    [lb, ub] = L_inf(flair(c,:,:), wm_mask(c, :,:), v1, v2);
                end
                [stats(c,:)] = computeStats(lb, ub, flair(c,:,:));
            
            end

            % toc(t);

            save("stats/" + transType + "_" + sbName + "_" + v1 + "_" + v2 + ".mat", "stats");

        end

    end
        

end




%% Helper Functions

% Get transformation bounds
function [img_lb, img_ub] = BiasField(img, order, coeffs, cRange)
% Get the code from torchIO to generate the BiasField
% https://torchio.readthedocs.io/_modules/torchio/transforms/augmentation/intensity/random_bias_field.html#RandomBiasField

    % Transform variables
    coeffs = str2double(coeffs);
    order  = str2double(order);
    cRange = str2double(cRange);

    img = squeeze(img);
    
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
    % img_diff = img_ub - img_lb;
    % V(:,:,:,1) = img_lb; % assume lb is center of set (instead of img)
    % V(:,:,:,2) = img_diff ; % basis vectors
    % C = [1; -1]; % constraints
    % d = [1; -1];
    % I = ImageStar(V, C, d, 0, 1); % input set

end

% Generate bias field for transformation
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
    img = squeeze(img);
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

% Get transofrm size stats
function stats = computeStats(lb, ub, flair)
    % [max_diff, mean_diff, median_diff, max_flair]
    % Ensure inputs are 2D
    lb = squeeze(lb); 
    ub = squeeze(ub); 
    flair = squeeze(flair);
    % initialize stats var
    stats = zeros(1,6);
    % Compute stats for whole slice
    img_diff = abs(lb - ub);
    stats(1) = max(img_diff, [], 'all');
    stats(2) = mean(img_diff, 'all');
    stats(3) = median(img_diff, 'all');
    stats(4) = max(flair, [], 'all'); % there is always some 0s, so this is good enough
    stats(5) = min(flair, [], 'all'); % there is always some 0s, so this is good enough
    stats(6) = stats(1)/abs(stats(4)-stats(5));

    % Compute stats for center patch of slice (64x64)
    % win1 = centerCropWindow2d(size(flair), [64 64]); % important features seems to be a bit shifted to the right
    % imS = imcrop(flair,win1);
    % imDiffS = imcrop(img_diff,win1);
    % xC = win1.XLimits; 
    % yC = win1.YLimits + 40; % something like this seems to be closer to the worst case scenarios
    % imDiffS = img_diff(xC(1):xC(2), yC(1):yC(2));
    % stats(5) = max(imDiffS, [], 'all');
    % stats(6) = mean(imDiffS, 'all');
    % stats(7) = median(imDiffS, 'all');

end

% Generate bounds for Linf
function [lb, ub, idxs] = L_inf(img, wm_mask, epsilon, nPix)

    rng(0); % to replicate results

    % Our Implementation
    epsilon = str2double(epsilon);
    nPix = str2double(nPix);

    wm_mask = squeeze(wm_mask);
    img = squeeze(img);

    % First get all the pixels in image that contains wm
    idxs = find(wm_mask == 1);
    N = length(idxs); % how many wm_pixels?
    cN = floor(N * nPix/100); % these are the total pixels to modify

    idxs = randperm(N,cN); % choose cN pixels out of N for each patch

    % Get data ranges
    img_min = min(img, [], 'all');
    img_max = max(img, [], 'all');
    img_range = img_max-img_min;

    epsilon = epsilon *img_range; % resize epsilon to match the equivalence of color values argued for

    % Apply perturnbation to those pixels
    lb = img;
    lb(idxs) = lb(idxs) - epsilon;
    ub = img;
    ub(idxs) = ub(idxs) + epsilon;

end

% Adjust contrast Perturbation
function [img1, img2] = AdjustContrast(img, gamma, gamma_range)

    % Our Implementation
    img = squeeze(img);
    img_min = min(img, [], 'all');
    img_max = max(img, [], 'all');
    img_range = img_max-img_min;
    epsilon = 1e-7; % not sure why we need this, but to avoid dividing by zero probably ???
    gamma = str2double(gamma);
    gamma_range = str2double(gamma_range);

    % Would this look something like...
    img1 = ((img-img_min)./(img_range+epsilon)).^(gamma-gamma_range) * img_range + img_min;
    img2 = ((img-img_min)./(img_range+epsilon)).^(gamma+gamma_range) * img_range + img_min;


end



