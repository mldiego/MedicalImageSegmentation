function [inference_metrics, verifiedGT_metrics, verifiedPred_metrics] = semantic_segmentation_metrics(gt, yPred, reachFile)
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
% Recall:
% TP : True Positives
% TN : True Negatives
% FP : False Positives
% FN : False Negatives
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

    load(reachFile); % 3 variables -> (R, rT, ME)
    
    if ~isempty(ME)
        % we do not like errors (ME = exception)
        warning("There is no output set compute, an error was encountered.");
    
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

        % 5) True positive rate (TPR)

        % 6) False Positive Rate (FPR)

        % 7) LTPR?

        % 8) LFPR?

        % 9) Volume difference (VD) ?

        % 10) Corr?

        % Extra: MATLB provided metrics for semantic segmentation
        % metrics = evaluateSemanticSegmentation(gt,yPred); % not sure what the format for this is to work
    
    end

end

%% Individual metric functions

% volume difference? VD?

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

% Get global robustness value / certified accuracy
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

% Get lesion-wise (region) global robustness value / certified accuracy
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
