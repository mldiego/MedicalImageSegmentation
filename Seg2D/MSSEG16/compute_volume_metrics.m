%% Get transform stats for msseg models given a bias field perturbation

% Study variables
% order = "3"; % possible polynomial order values ( > 1, default = 3)
% var1 = [0.1, 0.25, 0.5]; % coeff
% var2 = [0.00025, 0.0005, 0.001]; % coeff_range
% transType = "BiasField";

var1 = [0.002; 0.004; 0.006]; % (epsilon) equivalent to 1,2, 3 pixel color values
var2 = [5, 10, 15]; % (nPix)
transType = "linf"; % transformation type

% var1 = [0.5; 1; 2]; % (gamma)
% var2 = [0.0025; 0.00375; 0.005]; % gamma ranges to consider for each gamma value
% transType = "AdjustContrast";

% Data
path2data = "../../data/MSSEG16/subjects/";
% subjects = ["CHJE/1"];
subjects = ["CHJE/1", "DORE/1", "GULE/1"]; % subject data to analyze
% subjects = ["CHJE/1", "GULE/1"]; % subject data to analyze

models = [64, 80, 96];
% models = 64;

for m=1:length(models)

    sliceSize = models(m);

    % load model
    net = importONNXNetwork("models/size_"+string(sliceSize)+"/best_model.onnx");

    for s = 1:length(subjects)
    
        sb = subjects(s);
        sbName = split(sb, '/');
        sbName = string(sbName{1});
    
        % load 3d data
        flair = niftiread(path2data + sb+"/flair.nii");
        mask = double(niftiread(path2data + sb+"/mask.nii"));
        if strcmp(sbName, "DORE")
            flair = flair(:, :, 17:end-16); % remove 0s and make sure dims 2 and 3 are same
            mask = mask(:, :, 17:end-16);
        end
        flair = flair_normalization(flair);
            
        for j = 1:length(var1)
    
            v1 = var1(j);
            v1 = string(v1);
    
            for k = 1:length(var2)
            
                v2 = var2(k);
                v2 = string(v2);
                
                try

                    % Get predicted and verified volume data
                    [ver_vol, pred_vol,verTime] = recreate_volume(net, flair, sliceSize, sbName, transType, v1, v2);
                    % Compute metrics (predicted vs ground truth)
                    pred_metrics = volume_metrics(pred_vol, mask);
                    % Compute metrics (reachability vs ground truth)
                    gt_metrics = volume_metrics(ver_vol, mask);
                    % Compute metrics (reachability vs predicted)
                    robust_metrics = volume_metrics(ver_vol, pred_vol);
        
                    % Save data
                    save("metrics/"+sbName+"_"+string(sliceSize)+"_"+transType+"_"+string(v1)+"_"+string(v2)+".mat", ...
                        "robust_metrics","gt_metrics","pred_metrics","verTime");

                catch ME
                    % something failed, not sure why
                    warning(ME.message);
                    disp(sbName+"_"+string(sliceSize)+"_"+transType+"_"+string(v1)+"_"+string(v2));

                end
                
            end
    
        end
    
    end

end

