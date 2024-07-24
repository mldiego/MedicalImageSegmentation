% Study variables
% order = "3"; % possible polynomial order values ( > 1, default = 3)
% var1 = [0.1, 0.25, 0.5]; % coeff
% var2 = [0.00025, 0.0005, 0.001]; % coeff_range
% transType = "BiasField";

% var1 = [0.001; 0.002]; %
% var2 = [1,2,5]; % (nPix)
var1 = 0.001;
var2 = 1;
transType = "linf";

% var1 = [0.5; 1; 2]; % (gamma)
% var1 = [0.5; 1];
% var2 = [0.0025; 0.00375; 0.005]; % gamma ranges to consider for each gamma value
% transType = "AdjustContrast";

% Data
path2data = "../../data/ISBI/subjects/01/";
% subjects = ["01", "02", "03", "04"];
subjects = ["01"];


models = [64, 80, 96];
% models = 96;

for m=1:length(models)

    sliceSize = models(m);

    % load model
    net = importNetworkFromONNX("models/model"+string(sliceSize)+".onnx");

    for s = 1:length(subjects)
    
        sb = subjects(s);
        sbName = split(sb, '/');
        sbName = string(sbName{1});
    
        % load 3d data
        flair = niftiread(path2data + sb+"/flair.nii");
        mask = double(niftiread(path2data + sb+"/mask1.nii"));
        flair = flair_normalization(flair);
            
        for j = 1:length(var1)
    
            v1 = var1(j);
            v1 = string(v1);
    
            for k = 1:length(var2)
            
                v2 = var2(k);
                v2 = string(v2);
                
                % Get predicted and verified volume data
                [ver_vol, pred_vol,verTime] = recreate_volume(net, flair, sliceSize, sbName, transType, v1, v2);
                % Compute metrics (predicted vs ground truth)
                pred_metrics = volume_metrics(pred_vol, mask);
                % Compute metrics (reachability vs ground truth)
                gt_metrics = volume_metrics(ver_vol, mask);
                % Compute metrics (reachability vs predicted)
                robust_metrics = volume_metrics(ver_vol, pred_vol);
    
                % Save data
                save("metrics/ISBI_"+sbName+"_"+string(sliceSize)+"_"+transType+"_"+string(v1)+"_"+string(v2)+".mat", ...
                    "robust_metrics","gt_metrics","pred_metrics","verTime");

                % Create table for latex
                create_table_metrics(sbName, string(sliceSize), transType, string(v1), string(v2),...
                pred_metrics, gt_metrics, robust_metrics, verTime)
                
            end
    
        end
    
    end

end

