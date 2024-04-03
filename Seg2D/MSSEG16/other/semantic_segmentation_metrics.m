function [inference_metrics, verifiedGT_metrics, verifiedPred_metrics, rT] = semantic_segmentation_metrics(gt, yPred, reachFile)
% Compute all semantic segmentation metrics for a given reachability result
% These metrics are used in the ISBI challenge
%   - Dice Similarity Score           (DSC)  --> dice in MATLAB
%   - Positive Predictive Value       (PPV)  --> TP / (TP + FP)
%   - Lesion-wise True Positive Rate  (LTPR) --> TP / (TP + FN)
%   - Lesion-wise False Positive Rate (LFPR) --> FP / (FP + TN) (but lesion-wise only, see below for leasio-wise definition)
%   - Volume Difference               (VD)   --> absolute difference in volumes divided by the true volume (not sure if we will use this one)
%   - Volume Correlation              (Corr) --> Pearson's correlation coefficient
% For our case, the iages evaluated are 2D, so the last 2 may have to be
% converted to area difference and area correlation
% The overal score (SC) is given by the following combination of scores:
% SC = DSC/8 + PPV/8 + (1-LFPR)/4 + LTPR/4 + Corr/4;
%
% Notes:
% TP : True Positives  (lesion recognized as lesion)
% TN : True Negatives  (background recognized as background)
% FP : False Positives (background recognized as lesion)
% FN : False Negatives (lesion recognized as background)
%
% Lesion true positive rate (LTPR) is the lesion-wise ratio of true positives 
% to the sum of true positives and false negatives. We define the list of 
% lesions, ℒR, as the 18-connected components of ℳR and define ℒA in a similar manner.
% where ∣ℒR ∩ ℒA∣ counts any overlap between a connected component of ℳR and ℳL; 
% which means that both the human rater and algorithm have identified the same lesion, 
% though not necessarily having the same extents. 
% 
% Lesion false positive rate (LFPR) is the lesion-wise ratio of false positives 
% to the sum of false positives and true negatives,
%
% INPUTS:
%  - gt: Ground truth
%  - yPred: network's prediction on original image (no transform/perturbation)
%  - reachFile: reachability output for given image + transform
% 
% Other notes:
% Some of the metrics are NaN as they are 0/0. Let's ask people how they
% deal with this.

    load(reachFile); % 3 variables -> (R, rT, ME)
    
    if ~isempty(ME)
        % we do not like errors (ME = exception)
        warning("There is no output set computed, an error was encountered.");
    
        % Return empty scores
        inference_metrics    = [];
        verifiedGT_metrics   = [];
        verifiedPred_metrics = [];
    
    else
        
        % Initialize metrics
        inference_metrics    = struct;
        verifiedGT_metrics   = struct;
        verifiedPred_metrics = struct;
    
        % For faster verification, get bound estimates for each pixel
        [lb,ub] = R.estimateRanges; % overapproximation of actual set computed
        yPred = (yPred > 0);
    
        % 1) Get robustness value / certified accuracy (1 = correct, 0 = incorrect)
        inference_metrics.cra    = get_accuracy(yPred, gt);
        verifiedGT_metrics.cra   = get_accuracy(lb, ub, gt);
        verifiedPred_metrics.cra = get_accuracy(lb, ub, yPred);
        
        % 2) Get lesion-wise robustness value / certified accuracy
        inference_metrics.lcra    = get_lesionWise_accuracy(yPred, gt);
        verifiedGT_metrics.lcra   = get_lesionWise_accuracy(lb, ub, gt);
        verifiedPred_metrics.lcra = get_lesionWise_accuracy(lb, ub, yPred);

        % 3) Dice score (DSC)
        inference_metrics.dsc    = get_dice_score(yPred, gt);
        verifiedGT_metrics.dsc   = get_dice_score(lb, ub, gt);
        verifiedPred_metrics.dsc = get_dice_score(lb, ub, yPred);

        % 4) Positive Predictive Value (PPV)
        inference_metrics.ppv    = get_ppv(yPred, gt);
        verifiedGT_metrics.ppv   = get_ppv(lb, ub, gt);
        verifiedPred_metrics.ppv = get_ppv(lb, ub, yPred);

        % 5) True positive rate (TPR)
        inference_metrics.tpr    = get_tpr(yPred, gt);
        verifiedGT_metrics.tpr   = get_tpr(lb, ub, gt);
        verifiedPred_metrics.tpr = get_tpr(lb, ub, yPred);

        % 6) False Positive Rate (FPR)
        inference_metrics.fpr    = get_fpr(yPred, gt);
        verifiedGT_metrics.fpr   = get_fpr(lb, ub, gt);
        verifiedPred_metrics.fpr = get_fpr(lb, ub, yPred);

        % 7) LTPR?
        % inference_metrics.ltpr    = get_lesion_tpr(yPred, gt);
        % verifiedGT_metrics.ltpr   = get_lesion_tpr(lb, ub, gt);
        % verifiedPred_metrics.ltpr = get_lesion_tpr(lb, ub, yPred);

        % 8) LFPR?
        % inference_metrics.lfpr    = get_lesion_fpr(yPred, gt);
        % verifiedGT_metrics.lfpr   = get_lesion_fpr(lb, ub, gt);
        % verifiedPred_metrics.lfpr = get_lesion_fpr(lb, ub, yPred);

        % 9) Volume difference (VD) ?
        % inference_metrics.vd    = get_volume_difference(yPred, gt);
        % verifiedGT_metrics.vd   = get_volume_difference(lb, ub, gt);
        % verifiedPred_metrics.vd = get_volume_difference(lb, ub, yPred);

        % 10) Corr?
        % inference_metrics.corr    = get_volume_correlation(yPred, gt);
        % verifiedGT_metrics.corr   = get_volume_correlation(lb, ub, gt);
        % verifiedPred_metrics.corr = get_volume_correlation(lb, ub, yPred);

        % Extra: MATLAB provided metrics for semantic segmentation
        % metrics = evaluateSemanticSegmentation(gt,yPred); % not sure what the format for this is to work

        % 11) Compute Overall Score (SC)

    
    end

end

%% Individual metric functions

% Get the confusion matrix values

% Compute Volume Correlation? How?? (Corr)
function corr = get_volume_correlation(varargin)
    if nargin == 2 % (pred, target)
    
        pred = double(varargin{1});
        target = double(varargin{2});

    else

        lb = varargin{1};
        ub = varargin{2};
        target = varargin{3};
        
    end
end

% Compute volume difference? How?? (VD)
function vd = get_volume_difference(varargin)
    if nargin == 2 % (pred, target)
    
        pred = double(varargin{1});
        target = double(varargin{2});

    else

        lb = varargin{1};
        ub = varargin{2};
        target = varargin{3};
        
    end
end

% Compute Lesion-wise False Positive Rate (LFPR)
function lfpr = get_lfpr(varargin)
    if nargin == 2 % (pred, target)
    
        pred = double(varargin{1});
        target = double(varargin{2});

    else

        lb = varargin{1};
        ub = varargin{2};
        target = varargin{3};

    end
end

% Compute False Positive Rate (FPR) -> FP / (FP + TN)
function fpr = get_fpr(varargin)
    if nargin == 2 % (pred, target)
    
        pred = double(varargin{1});
        target = double(varargin{2});

        C = confusionmat(pred(:), target(:)); % [TN, FN; FP TP]
        C = postProcessCMat(C);
        fpr = C(2,1) / (C(2,1) + C(1,1));

    else

        lb = varargin{1};
        ub = varargin{2};
        target = varargin{3};


        % 1) get correctly classified as 0 (background)
        ver_background = (ub <= 0);
        ver_background = ~ver_background;
        ver_background = (ver_background == target) & (ver_background == 0);
        
        % 2) get correctly classified as 1 (lession)
        ver_lession = (lb > 0);
        ver_lession = (ver_lession == target) & (ver_lession == 1);
    
        % 3) Get all correctly verified pixels
        ver_img = ver_background + ver_lession; % could refine this by looking at the exact reach sets computed (reach set vs halfspace)
        
        % 4) Compute score
        C = confusionmat(single(ver_img(:)), single(target(:))); % [TN, FN; FP TP]
        C = postProcessCMat(C);
        fpr = C(2,1) / (C(2,1) + C(1,1));

    end
end

% Compute Lesion-wise True Positive Rate (LTPR)
function ltpr = get_ltpr(varargin)
    if nargin == 2 % (pred, target)
    
        pred = double(varargin{1});
        target = double(varargin{2});

    else

        lb = varargin{1};
        ub = varargin{2};
        target = varargin{3};

    end
end

% Compute True Positive Rate (TPR) -> TP / (TP + FN)
function tpr = get_tpr(varargin)
    if nargin == 2 % (pred, target)
    
        pred = double(varargin{1});
        target = double(varargin{2});
        
        C = confusionmat(pred(:), target(:)); % [TN, FN; FP TP]
        C = postProcessCMat(C);
        tpr = C(2,2) / (C(2,2) + C(1,2));

    else

        lb = varargin{1};
        ub = varargin{2};
        target = varargin{3};


        % 1) get correctly classified as 0 (background)
        ver_background = (ub <= 0);
        ver_background = ~ver_background;
        ver_background = (ver_background == target) & (ver_background == 0);
        
        % 2) get correctly classified as 1 (lession)
        ver_lession = (lb > 0);
        ver_lession = (ver_lession == target) & (ver_lession == 1);
    
        % 3) Get all correctly verified pixels
        ver_img = ver_background + ver_lession; % could refine this by looking at the exact reach sets computed (reach set vs halfspace)
        
        % 4) Compute score
        C = confusionmat(single(ver_img(:)), single(target(:))); % [TN, FN; FP TP]
        C = postProcessCMat(C);
        tpr = C(2,2) / (C(2,2) + C(1,2));

    end
end

% Compute positive predictive value (PPV) -> TP / (TP + FP)
function ppv = get_ppv(varargin)
    if nargin == 2 % (pred, target)
    
        pred = double(varargin{1});
        target = double(varargin{2});

        C = confusionmat(pred(:), target(:)); % [TN, FN; FP TP]
        C = postProcessCMat(C);
        ppv = C(2,2) / (C(2,2) + C(2,1)); 

    else

        lb = varargin{1};
        ub = varargin{2};
        target = varargin{3};


        % 1) get correctly classified as 0 (background)
        ver_background = (ub <= 0);
        ver_background = ~ver_background;
        % ver_background = (ver_background == target) & (ver_background == 0);
        
        % 2) get correctly classified as 1 (lession)
        ver_lesion = (lb > 0);
        % ver_lession = (ver_lession == target) & (ver_lession == 1);
    
        % 3) get verified output
        ver_img = 2*ones(size(target)); % pixel = 2 -> unknown
        
        background = find(ver_background == 0); % 0
        ver_img(background) = 0;
        
        lesion = find(ver_lesion == 1); % 1
        ver_img(lesion) = 1;


        % 4) Compute score
        C = confusionmat(single(ver_img(:)), single(target(:))); % [TN, FN; FP TP]
        C = postProcessCMat(C);
        ppv = C(2,2) / (C(2,2) + C(2,1)); 

    end
end

% Ensure confusion matrix is the right size
function C = postProcessCMat(C)

    if size(C) == [1,1] % all are TN (only happens with predictions)
        cm = zeros(2,2);
        cm(1) = C;
        C = cm;
    elseif size(C) == [3,3] % there are some unknowns (2)
        % Column 3 will always be 0 (no 2 in gt)
        % C(3,1) -> verified as 2, originally 0 (assume this if FP)
        % C(3,2) -> verified as 2, originally 1 (assume this is FN)
        cm = C(1:2, 1:2);
        cm(2,1) = cm(2,1) + C(3,1); % FP
        cm(1,2) = cm(1,2) + C(3,2); % FN
        C = cm;
    elseif size(C) == [2,2] % 0s and 1s, as expected
        "all good";
    else
        disp(C);
        error("Confusion matrix is something else");
    end

end

% Compute dice similarity score (DCS) (see: doc dice)
function dsc = get_dice_score(varargin)

    if nargin == 2 % (pred, target)

        pred = double(varargin{1});
        target = double(varargin{2});

        dsc = dice(pred,target);

    else

        lb = varargin{1};
        ub = varargin{2};
        target = varargin{3};

        % 1) get correctly classified as 0 (background)
        ver_background = (ub <= 0);
        ver_background = ~ver_background;
        % ver_background = (ver_background == target) & (ver_background == 0);
        
        % 2) get correctly classified as 1 (lession)
        ver_lesion = (lb > 0);
        % ver_lesion = (ver_lesion == target) & (ver_lesion == 1);
    
        % 3) Get all correctly verified pixels
        % ver_img = ver_background + ver_lession; % could refine this by looking at the exact reach sets computed (reach set vs halfspace)
    
        % 3) get verified output
        ver_img = 2*ones(size(target)); % pixel = 2 -> unknown
        
        background = find(ver_background == 0); % 0
        ver_img(background) = 0;
        
        lesion = find(ver_lesion == 1); % 1
        ver_img(lesion) = 1;
        
        % 4) compute dice score
        dsc = dice(double(target), double(ver_img));
        dsc = dsc(~isnan(dsc));
        dsc = sum(dsc)/length(dsc);

    end
        

end

% Get global robustness value / certified accuracy (CRA)
function cra = get_accuracy(varargin)

    if nargin == 2 % (pred, target)

        pred = varargin{1};
        target = varargin{2};

        cra = pred == target;
        cra = sum(cra, 'all')/numel(cra);

    else

        lb = varargin{1};
        ub = varargin{2};
        target = varargin{3};

        % 1) get correctly classified as 0 (background)
        ver_background = (ub <= 0);
        ver_background = ~ver_background;
        ver_background = (ver_background == target) & (ver_background == 0);
        
        % 2) get correctly classified as 1 (lession)
        ver_lession = (lb > 0);
        ver_lession = (ver_lession == target) & (ver_lession == 1);
    
        % 3) Get all correctly verified pixels
        ver_img = ver_background + ver_lession; % could refine this by looking at the exact reach sets computed (reach set vs halfspace)
    
        % 4) Robustness value (certified accuracy, pixel-wise)
        cra = sum(ver_img, 'all')/numel(ver_img);

    end

end

% Get lesion-wise (region) global robustness value / certified accuracy (LCRA)
function cra = get_lesionWise_accuracy(varargin)

    if nargin == 2 % (pred, target)

        pred = varargin{1};
        target = varargin{2};
        
        cra = (pred == target) & (target == 1);
        cra = sum(cra, "all")/sum(target == 1, "all");

    else

        lb = varargin{1};
        ub = varargin{2};
        target = varargin{3};

        % 1) get correctly classified as 0 (background)
        ver_background = (ub <= 0);
        ver_background = ~ver_background;
        ver_background = (ver_background == target) & (ver_background == 0);
        
        % 2) get correctly classified as 1 (lession)
        ver_lession = (lb > 0);
        ver_lession = (ver_lession == target) & (ver_lession == 1);
    
        % 3) Get all correctly verified pixels
        ver_img = ver_background + ver_lession; % could refine this by looking at the exact reach sets computed (reach set vs halfspace)

        % 4) Pixels correctly classified in region of interest
        % only care about lession (label = 1 region)
        cra = (ver_img == 1) & (target == 1);
        cra = sum(cra, "all")/sum(target==1, "all");

    end

end
