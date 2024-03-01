%% Verify msseg models given a bias field perturbation

% rng(0);

% Study variables
sliceSizes = [64, 80, 96]; % for cropping and loading models
order = 3; % possible polynomial order values ( > 1, default = 3)
coeff = [0.1, 0.25, 0.5];
coeff_range = [0.00025, 0.0005, 0.001]; % what should the size of this be? 
path2data = "../data/MSSEG16/subjects/";
subjects = ["CHJE/1"]; % subject data to analyze (from test, Han will provide)
transType = "BiasField";

% Perturbation to evaluate
% transform = struct;
% transform.name = "BiasField";
% transform.order = order;
% transform.coefficient = coeff;
% transform.coefficient_range = coeff_range;


%% Reachability analysis for all models

% Define reachability options
% reachOptions = struct;
reachMethod = 'relax-star-range';
relaxFactor = 0.95;
% reachOptions.reachMethod = 'approx-star';

for s = 1:length(subjects)

    % load 3d data
    flair   = niftiread(path2data + subjects(s)+"/flair.nii");
    mask    = niftiread(path2data + subjects(s)+"/mask.nii");
    wm_mask = niftiread(path2data + subjects(s)+"/wm_mask.nii");
    [flair, mask, wm_mask] = removeExtraBackground(flair, mask, wm_mask);

    % Begin exploration
    for i=1:length(sliceSizes)

        sZ = sliceSizes(i);
        
        for j = 1:length(coeff)
    
            coefficient = coeff(j);
    
            for k = 1:length(coeff_range)
            
                cRange = coeff_range(k);

                for c = 1:size(flair,1) % iterate through all 2D slices

                    generate_patches(flair, mask, wm_mask, sZ, c); % generates all possible patches to analyze

                    patches = dir("tempData/*.mat"); % get generated patches

                    for p = 1:height(patches)

                        img_path = "sliceData"+patches(p).name;
    
                        verify_model_subject_patch(img_path, sZ, reachMethod, relaxFactor, transType, order, coefficient, cRange);
                    
                    end

                    delete("tempData/*.mat"); % remove all generated patches
                
                end

            end
    
        end
        
    end

end



% This works, but use the "system" command to pass arguments to the
% functions;
% ! timeout 5 matlab -r "disp('Running matlab'); pause(10); quit"

%% Helper Functions

function [flair, mask, wm_mask] = removeExtraBackground(flair, mask, wm_mask)
    
    % Remove all background around brain to simplify the verification process
    % This reduces the number of slices to verify
    
    % First remove all 2D slices that are just background
    cropMask = flair < 3;% Find all zeros, even those inside the image.
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
function slice_points = generate_patches(flair, mask, wm_mask, sZ, c)

    % generate bounds for the whole slize given bias field transformation
    % using the 3D data info or 2D data? 2D data for now
    xC = 1:sZ:size(flair,2); % initialize window coordinates
    yC = 1:sZ:size(flair,3); 
    for i = xC
        for j = yC
            flair = flair(c, i:i+sZ, j:j+sZ);
            mask = mask(c, i:i+sZ, j:j+sZ);
            wm_mask = mask(c, i:i+sZ, j:j+sZ);
            if all(wm_mask == 0, 'all') % do not evaluate unless white matter in it
                continue % no need to do this one
            else
                saveName = fprintf("tempData/data_%s_%s_%s.mat", c, i, j);
                save(saveName, "flair", "mask", "wm_mask", "lb", "ub");
            end
        end
    end


end


