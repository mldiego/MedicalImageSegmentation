%% Verify msseg models given a bias field perturbation

% Study variables
sliceSizes = [64, 80, 96]; % for cropping and loading models
% epsilon = [0.002; 0.004; 0.006]; % equivalent to 1, 2, 3 pixel color values
epsilon = 0.002;
nPix = 5; % [0.5, 1]
% nPix = [1, 2, 5]; % percentage of pixels within white matter mask perturbed
path2data = "../../data/UMCL/subjects/patient";
% subjects = ["01", "02", "03", "04", "05", "06"]; % subject data to analyze (only use mask1 for each)
subjects = ["01"];
transType = "linf";


%% Reachability analysis for all models

% Define reachability options
% reachOptions = struct;
reachMethod = "relax-star-range";
relaxFactor = "1";
% reachOptions.reachMethod = 'approx-star';

% parCluster = gcp('nocreate');
% parpool(4); % less than max to avoid out of memory errors

for s = 1:length(subjects)

    sb = subjects(s);
    sbName = sb;

    % load 3d data
    flair   = niftiread(path2data + sb+"/1/flair.nii");
    flair = flair_normalization(flair);
    mask    = niftiread(path2data + sb+"/1/mask.nii");
    wm_mask = niftiread(path2data + sb+"/1/wm_mask.nii");

    % Begin exploration
    for i = 1:length(sliceSizes)

        sZ = sliceSizes(i);
        sZ = string(sZ);
        
        for j = 1:length(epsilon)
    
            ep = epsilon(j);
            ep = string(ep);
    
            for k = 1:length(nPix)
            
                nP = nPix(k);
                nP = string(nP);

                disp("Verifying subject " + sbName +", model " + sZ + ", epsilon "+ ep +", and "+ nP +"% of pixels");
                t = tic;

                % parfor
                for c = 1:size(flair,1) % iterate through all 2D slices 

                    generate_patches(flair, mask, wm_mask, sZ, c, ep, nP); % generates all possible patches to analyze

                    patches = dir("tempData/data_"+string(c)+"_*.mat"); % get generated patches

                    npatch = height(patches);

                    disp("  - Checking slice "+string(c) +" with " + string(npatch)+" patches");

                    for p = 1:npatch

                        img_path = "tempData/"+patches(p).name;

                        verify_model_subject_patch(img_path, sb, sZ, reachMethod, relaxFactor, transType, ep, nP);
                        
                    end

                    delete("tempData/data_"+string(c)+"_*.mat"); % remove all generated patches
                
                end

                toc(t);
                
                % shut down parpool (not closing it may lead to errors)
                poolobj = gcp('nocreate');
                delete(poolobj);

            end
    
        end
        
    end

end




%% Helper Functions

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
    [flair_lb, flair_ub, idxs] = L_inf(flair_slice, wm_mask_slice, epsilon, nPix);

    save("linfData/patch"+string(c)+".mat", "idxs");

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

    id = randperm(N,cN); % choose cN pixels out of N for each patch
    idxs = idxs(id);

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



