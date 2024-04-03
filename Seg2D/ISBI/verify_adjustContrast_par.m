%% Verify msseg models given a bias field perturbation

% Study variables
sliceSizes = [64, 80, 96]; % for cropping and loading models
gamma = [0.5; 1; 2]; % lower and upper bound for typical ranges used for gamma
gamma_range = [0.0025; 0.00375; 0.005]; % gamma ranges to consider for each gamma value
path2data = "../../data/ISBI/subjects/01/";
subjects = ["01", "02", "03", "04"]; % subject data to analyze (omly use mask1 for each)
transType = "AdjustContrast";


%% Reachability analysis for all models

% Define reachability options
% reachOptions = struct;
reachMethod = "relax-star-range";
relaxFactor = "0.95";
% reachOptions.reachMethod = 'approx-star';

for s = 1:length(subjects)

    sb = subjects(s);
    sbName = split(sb, '/');
    sbName = string(sbName{1});

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
        
        for j = 2:length(gamma)
    
            gval = gamma(j);
            gval = string(gval);
    
            for k = 1:length(gamma_range)
            
                gRange = gamma_range(k);
                gRange = string(gRange);

                parfor c = 1:size(flair,1) % iterate through all 2D slices 

                    generate_patches(flair, mask, wm_mask, sZ, c, gval, gRange); % generates all possible patches to analyze

                    patches = dir("tempData/data_"+string(c)+"_*.mat"); % get generated patches

                    for p = 1:height(patches)

                        img_path = "tempData/"+patches(p).name;
                        verify_model_subject_patch(img_path, sb, sZ, reachMethod, relaxFactor, transType, gval, gRange);

                    end

                    delete("tempData/data_"+string(c)+"_*.mat"); % remove all generated patches
                
                end

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
function generate_patches(flair, mask, wm_mask, sZ, c, gamma, gRange)

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
    [flair_lb, flair_ub] = AdjustContrast(flair_slice, gamma, gRange);

    width = size(flair_slice,2);
    height = size(flair_slice,1);

    for i = xC
        for j = yC

            % Get data info
            flair = flair_slice(i:min(i+sZ-1, height), j:min(j+sZ-1, width));
            mask = mask_slice(i:min(i+sZ-1, height), j:min(j+sZ-1, width));
            wm_mask = wm_mask_slice(i:min(i+sZ-1, height), j:min(j+sZ-1, width));
            
            % Get bounds for that data
            lb = flair_lb(i:min(i+sZ-1, height), j:min(j+sZ-1, width));
            ub = flair_ub(i:min(i+sZ-1, height), j:min(j+sZ-1, width));
            
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

% Adjust contrast Perturbation
function [img1, img2] = AdjustContrast(img, gamma, gamma_range)
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
    % if gamma == 2 % upper range
    %     gamma = gamma - gamma_range;
    % end
    % The range for gamma is 0.5 to 2 as default for most code I have seen
    
    % This is the transformed image. 
    % img_trans = ((img-img_min)./(img_range+epsilon)).^gamma * img_range + img_min;

    % Do we create the set from here by assuiming gama is a set of values?
    % This may make the most sense...
    % I believe this would assign no range values for the background
    % 
    % Would this look something like...
    img1 = ((img-img_min)./(img_range+epsilon)).^(gamma-gamma_range) * img_range + img_min;
    img2 = ((img-img_min)./(img_range+epsilon)).^(gamma+gamma_range) * img_range + img_min;
    % IS = ImageStar(lb,ub);
    
    % However, this will not be computationally efficient (defintely not as
    % efficient as the bright/dark perturbations)
    
    % Would this work?

    % Define input set as ImageStar
    % img_diff = img1 - img2;
    % V(:,:,:,1) = img2; % assume lb is center of set (instead of img)
    % V(:,:,:,2) = img_diff ; % basis vectors
    % C = [1; -1]; % constraints
    % d = [1; -1];
    % I = ImageStar(V, C, d, 0, 1); % input set

    % Is this okay? Check point
    % [lb1, ub1] = I.estimateRanges;
    % lb_diff = lb1 - img2;
    % ub_diff = ub1 - img1;

end



