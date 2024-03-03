%% Verify msseg models given a bias field perturbation

% Study variables
sliceSizes = [64, 80, 96]; % for cropping and loading models
epsilon = [0.002; 0.004; 0.006]; % equivalent to 1,2, 3 pixel color values
nPix = [5, 10, 15];
path2data = "../../data/ISBI/subjects/01/";
subjects = ["01", "02", "03", "04"]; % subject data to analyze (omly use mask1 for each)
transType = "linf";


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
        
        for j = 1:length(gamma)
    
            gval = gamma(j);
            gval = string(gval);
    
            for k = 1:length(gamma_range)
            
                gRange = gamma_range(k);
                gRange = string(gRange);

                for c = 1:size(flair,1) % iterate through all 2D slices (it broke at 70, restart from there)

                    generate_patches(flair, mask, wm_mask, sZ, c, gval, gRange); % generates all possible patches to analyze

                    patches = dir("tempData/*.mat"); % get generated patches

                    for p = 1:height(patches)

                        img_path = "tempData/"+patches(p).name;

                        % sys_cmd = sprintf('C:/"Program Files"/Git/git-bash.exe timeout 450 matlab -r "cd ../../nnv/code/nnv; startup_nnv; cd ../../../MedicalImageSegmentation/Seg2D; verify_model_subject_patch(%s, %s, %s, %s, %s, %s, %s, %s, %s); quit;"', img_path, sbName, sZ, reachMethod, relaxFactor, transType, order, coefficient, cRange);
                        % sys_cmd = sprintf('C:/"Program Files"/Git/usr/bin/timeout.exe 450 matlab -r "addpath(genpath(''../../nnv/code/nnv'')); verify_model_subject_patch(""%s"", ""%s"", ""%s"", ""%s"", ""%s"", ""%s"", ""%s"", ""%s"", ""%s""); pause(0.5); quit force;"', img_path, sbName, sZ, reachMethod, relaxFactor, transType, order, coefficient, cRange);
                        sys_cmd = sprintf('timeout 450 matlab -r "addpath(genpath(''../../../nnv/code/nnv'')); verify_model_subject_patch(''%s'', ''%s'', ''%s'', ''%s'', ''%s'', ''%s'', ''%s'', ''%s''); pause(0.5); quit force;"', img_path, sbName, sZ, reachMethod, relaxFactor, transType, epsilon, nPix);

                        [status, cmdout] = system(sys_cmd);
                        % verify_model_subject_patch(img_path, sb, sZ, reachMethod, relaxFactor, transType, order, coefficient, cRange);
                        
                        % system("sleep 5"); % wait a few seconds to ensure matlab is closed (windows)

                    end

                    delete("tempData/*.mat"); % remove all generated patches
                
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
function generate_patches(flair, mask, wm_mask, sZ, c, epsilon, nPix)

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
    [flair_lb, flair_ub, idxs] = L_inf(flair_slice, epsilon, nPix);

    save("linfData/patch"+string(c)+".mat", "idxs");

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

% L_inf on randm pixels of each 2D patch
function [lb, ub, idxs] = L_inf(img, wm_mask, epsilon, nPix)

    rng(0); % to replicate results

    % Our Implementation
    epsilon = str2double(epsilon);
    nPix = str2double(nPix);

    % First get all the pixels in image that contains wm
    idxs = find(wm_mask == 1);
    N = length(idxs); % how many wm_pixels?
    cN = floor(N * nPix/100); % these are the total pixels to modify

    idxs = randperm(N,cN); % choose cN pixels out of N for each patch

    % Apply perturnbation to those pixels
    lb = img;
    lb(idxs) = lb(idxs) - epsilon;
    ub = img;
    ub(idxs) = ub(idxs) + epsilon;

end



