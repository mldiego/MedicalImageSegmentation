function metrics = volume_metrics(pred, gt)
% Compute all semantic segmentation metrics for a recomputed volume given verified output
% These metrics are used in the ISBI challenge
%   - Dice Similarity Score           (DSC)  --> dice in MATLAB
%   - Positive Predictive Value       (PPV)  --> TP / (TP + FP)
%   - Lesion-wise True Positive Rate  (LTPR) --> TP / (TP + FN)
%   - Lesion-wise False Positive Rate (LFPR) --> FP / (FP + TN) (but lesion-wise only, see below for leasio-wise definition)
%   - Volume Difference               (VD)   --> absolute difference in volumes divided by the true volume (not sure if we will use this one)
%   - Volume Correlation              (Corr) --> Pearson's correlation coefficient
%
%
% INPUTS:
%  - gt: Ground truth
%  - yPred: network's prediction on original image (no transform/perturbation)



    % Compute the sum of elements in predictions and ground truth
    pred_total = double(sum(pred(:)));
    gt_total = double(sum(gt(:)));

    % True positives
    tp = sum(pred(gt == 1));

    % DICE coefficient
    dice = 2 * tp / (pred_total + gt_total);

    % Positive predictive value (PPV) also known as precision
    ppv = tp / (pred_total + 0.001);

    % True positive rate (TPR) also known as recall
    tpr = tp / (gt_total + 0.001);

    % Volumetric difference
    vd = abs(pred_total - gt_total) / gt_total;

    % Calculate LFPR (Local False Positive Rate)
    pred_labels = bwlabeln(pred, 18);  % Use 18-connectivity
    pred_num = max(pred_labels(:));

    lfp_cnt = 0;
    for label = 1:pred_num
        if sum(gt(pred_labels == label)) == 0
            lfp_cnt = lfp_cnt + 1;
        end
    end
    lfpr = lfp_cnt / (pred_num + 0.001);

    % Calculate LTPR (Local True Positive Rate)
    gt_labels = bwlabeln(gt, 18);  % Use 18-connectivity
    gt_num = max(gt_labels(:));

    ltp_cnt = 0;
    for label = 1:gt_num
        if sum(pred(gt_labels == label)) > 0
            ltp_cnt = ltp_cnt + 1;
        end
    end
    ltpr = ltp_cnt / gt_num;

    % Calculate Pearson's correlation coefficient
    corr = corrcoef(pred(:), gt(:));
    corr = corr(1,2);

    % Combined score (sc)
    sc = (1/8)*dice + (1/8)*ppv + (1-lfpr)/4 + ltpr/4 + corr/4;

    % Create structure for metrics
    metrics = struct('dice', dice, 'ppv', ppv, 'tpr', tpr, 'lfpr', lfpr, ...
                     'ltpr', ltpr, 'vd', vd, 'corr', corr, 'sc', sc);
                     
end