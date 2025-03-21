%% Verify msseg models given a bias field perturbation

% Study variables
sliceSizes = [64, 80, 96]; % for cropping and loading models
order = "3"; % possible polynomial order values ( > 1, default = 3)
coeff = [0.1, 0.25, 0.5];
coeff_range = [0.00025, 0.0005, 0.001]; % what should the size of this be? 
path2data = "../../data/ISBI/subjects/01/";
subjects = ["01", "02", "03", "04"]; % subject data to analyze (omly use mask1 for each)
transType = "BiasField";


%% Reachability analysis for all models

% Define reachability options
% reachOptions = struct;
reachMethod = "relax-star-range";
relaxFactor = "1";
% reachOptions.reachMethod = 'approx-star';

for s = 1:length(subjects)

    sb = subjects(s);
    % sbName = split(sb, '/');
    % sbName = string(sbName{1});
    sbName = sb;

    % load 3d data
    flair   = niftiread(path2data + sb+"/flair.nii");
    flair = flair_normalization(flair);
    mask    = niftiread(path2data + sb+"/mask1.nii");
    wm_mask = niftiread(path2data + sb+"/wm_mask.nii");
    [flair, mask, wm_mask] = removeExtraBackground(flair, mask, wm_mask);

    % Begin exploration
    for i=1:length(sliceSizes)

        sZ = sliceSizes(i);
        sZ = string(sZ);
        
        for j = 1:length(coeff)
    
            coefficient = coeff(j);
            coefficient = string(coefficient);
    
            % for k = 1:length(coeff_range)
            
                % cRange = coeff_range(k);
                % cRange = string(cRange);

            for c = 1:size(flair,1) % iterate through all 2D slices (it broke at 70, restart from there)

                generate_patches(flair, mask, wm_mask, sZ, c, order, coefficient); % generates all possible patches to analyze

                patches = dir("tempData/data_"+string(c)+"_*.mat"); % get generated patches

                for p = 1:height(patches)

                    img_path = "tempData/"+patches(p).name;

                    verify_model_subject_patch(img_path, sbName, sZ, reachMethod, relaxFactor, transType, order, coefficient); 

                end

                delete("tempData/data_"+string(c)+"_*.mat"); % remove all generated patches
                

            end
    
        end
        
    end

end




%% Helper Functions

% Remove background (all useless 0s outside of brain)
function [flair, mask, wm_mask] = removeExtraBackground(flair, mask, wm_mask)
    
    % Remove all background around brain to simplify the verification process
    % This reduces the number of slices to verify
    
    % First remove all 2D slices that are just background
    cropMask = wm_mask < 1;% Find all zeros, even those inside the image.
    cropMask = imfill(cropMask, 'holes'); % Get rid of zeros inside image.
    % Invert mask and get bounding box.
    props = regionprops(~cropMask, 'BoundingBox');
    if ~isempty(props) % there may be nothing to crop out
        bvol = floor(props.BoundingBox); % bounding volume to crop around brain
        % Crop data
        flair = imcrop3(flair, bvol);
        mask = imcrop3(flair, bvol);
        wm_mask = imcrop3(wm_mask, bvol);
    end
    
end

% Generate starting points for slices
function generate_patches(flair, mask, wm_mask, sZ, c, order, coeffs)

    % generate bounds for the whole slize given bias field transformation
    % using the 3D data info or 2D data? 2D data for now
    sZ = str2double(sZ);
    xC = 1:sZ:size(flair,2); % initialize window coordinates
    if xC(end) > size(flair,2)
        xC = xC(1:end-1);
    end
    yC = 1:sZ:size(flair,3);
    if xC(end) > size(flair,2)
        xC = xC(1:end-1);
    end

    flair_slice   = squeeze(flair(c,:,:));
    mask_slice    = squeeze(mask(c,:,:));
    wm_mask_slice = squeeze(wm_mask(c,:,:));

    % Apply transformation
    [flair_lb, flair_ub] = BiasField(flair_slice, order, coeffs);

    width = size(flair_slice,2);
    height = size(flair_slice,1);

    for i = xC
        for j = yC

            % Get data info
            flair = flair_slice(min(i:i+sZ-1, height), min(j:j+sZ-1, width));
            mask = mask_slice(min(i:i+sZ-1, height), min(j:j+sZ-1, width));
            wm_mask = wm_mask_slice(min(i:i+sZ-1, height), min(j:j+sZ-1, width));
            
            % Get bounds for that data
            lb = flair_lb(min(i:i+sZ-1, height), min(j:j+sZ-1, width));
            ub = flair_ub(min(i:i+sZ-1, height), min(j:j+sZ-1, width));
            
            % Check the patches are of correct dimensions, otherwise add 0s
            % to the bottom/right
            if size(lb, 1) ~= sZ
                lb = padarray(lb, sZ-size(lb,1), 0, "post"); % add padding to the right
                ub = padarray(ub, sZ-size(ub,1), 0, "post"); % add padding to the right
            end
            if size(lb, 2) ~= sZ
                lb = padarray(lb, [0 sZ-size(lb,1)], 0, "post"); % add padding to the bottom
                ub = padarray(ub, [0 sZ-size(ub,1)], 0, "post"); % add padding to the bottom
            end
            
            % If slice is boring omit it
            if all(wm_mask == 0, 'all') % do not evaluate unless white matter in it
                continue % no need to do this one
            else % save data to analyze
                saveName = sprintf("tempData/data_%s_%s_%s.mat", string(c), string(i), string(j));
                save(saveName, "flair", "mask", "wm_mask", "lb", "ub");
            end
        end
    end


end

% Get transformation bounds
function [img_lb, img_ub] = BiasField(img, order, coeffs)
% Get the code from torchIO to generate the BiasField
% https://torchio.readthedocs.io/_modules/torchio/transforms/augmentation/intensity/random_bias_field.html#RandomBiasField

    % Transform variables
    coeffs = str2double(coeffs);
    order  = str2double(order);
    % cRange = str2double(cRange);
    
    % Field applied only to the center of the image (not background)
    img1 = img;
    bField1 = generate_bias_field(img, coeffs, order);
    img1(19:end-18,:) = img(19:end-18,:) .* bField1;

    % img2 = img;
    % bField2 = generate_bias_field(img, coeffs+cRange, order);
    % img2(19:end-18,:) = img(19:end-18,:) .* bField2;

    % get max and min value for every pixel given the biasField applied
    % interval range for every pixel is given by min and max values for
    % that pixel in images img1 and img2
    % img_lb = min(img1,img2);
    % img_ub = max(img1,img2);

    img_lb = img;
    img_ub = img1;

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
    shape = size(img); 
    shape(1) = shape(2); % copy lower dimension
    % shape = shape(2:end); % first axis is channels (not necessary as it is a greyscale image -> 1 channel, already removed dimension)
    half_shape = shape ./ 2;
    ranges = {};
    
    i = 1;
    for n = half_shape
        ranges{i} = (-n:1:(n-1)) + 0.5;
        i = i + 1;
    end
    
    bias_field = zeros(shape);
    
    ndim = length(shape);
    % meshes = zeros([ndim, shape(1), shape(2)]);
    meshes = {};
    for k=1:ndim
        meshes{k} = meshgrid(ranges{k});
    end
    
    for i = 1:length(meshes)
        mesh = meshes{i};
        mesh_max = max(mesh, [], 'all');
        if mesh_max > 0
            mesh = mesh./mesh_max;
            meshes{i} = mesh;
        end
    end
    
    x_mesh = meshes{1};
    y_mesh = meshes{2};
    
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

