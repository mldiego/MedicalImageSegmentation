%% Get transform stats for msseg models given a bias field perturbation

% Study variables
% order = "3"; % possible polynomial order values ( > 1, default = 3)
% var1 = [0.1, 0.25, 0.5]; % coeff
% var2 = [0.00025, 0.0005, 0.001]; % coeff_range
% transType = "BiasField";

var1 = [0.002; 0.004; 0.006]; % (epsilon) equivalent to 1,2, 3 pixel color values
var2 = [5, 10, 15]; % (nPix)
transType = "linf";

% var1 = [0.5; 1; 2]; % (gamma)
% var2 = [0.0025; 0.00375; 0.005]; % gamma ranges to consider for each gamma value
% transType = "AdjustContrast";

% Data
path2data = "../../data/MSSEG16/subjects/";
subjects = "CHJE/1";
% subjects = ["CHJE/1", "DORE/1", "GULE/1"]; % subject data to analyze
% subjects = ["DORE/1", "GULE/1"]; % subject data to analyze


for s = 1:length(subjects)

    sb = subjects(s);
    sbName = split(sb, '/');
    sbName = string(sbName{1});

    % load 3d data
    flair = niftiread(path2data + sb+"/flair.nii");
    wm_mask = niftiread(path2data + sb+"/mask.nii");
    % if strcmp(sbName, "DORE")
    %     flair = flair(:, :, 17:end-16); % remove 0s and make sure dims 2 and 3 are same
    %     wm_mask = wm_mask(:, :, 17:end-16);
    % end
    flair = flair_normalization(flair);
        
    for j = 1:length(var1)

        v1 = var1(j);
        v1 = string(v1);

        for k = 1:length(var2)
        
            v2 = var2(k);
            v2 = string(v2);

            [ver_vol, pred_vol] = recreate_volume(flair, sbName, transType, v1, v2);
            pred_metrics = volume_metrics
            [metrics] = semantic_segmentation_metrics(ver_vol, pred_vol, mask);

        end

    end
        

end



