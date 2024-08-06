% Compute average metrics for each model + transformation across timepoints

var1 = [0.001, 0.002, 0.004]; %
var2 = [1, 2, 5, 10]; % (nPix)
transType = "linf";

% var1 = ["0.5"; "1"; "2"]; % (gamma)
% var2 = "0.0025";
% transType = "AdjustContrast";

models = ["64", "80", "96"];
% models = ["64", "80"];

% Data
path2data = "../../data/ISBI/subjects/01/";
subjects = ["01", "02", "03", "04"];


for i = 1:length(var1)
    for j = 1:length(var2)
        for m = 1:length(models)
            disp("metrics/ISBI_"+models(m)+"_"+transType+"_"+var1(i)+"_"+var2(j));
            try
                % Load metrics
                m1 = load("metrics/ISBI_"+subjects(1)+"_"+models(m)+"_"+transType+"_"+var1(i)+"_"+var2(j)+".mat");
                m2 = load("metrics/ISBI_"+subjects(2)+"_"+models(m)+"_"+transType+"_"+var1(i)+"_"+var2(j)+".mat");
                m3 = load("metrics/ISBI_"+subjects(3)+"_"+models(m)+"_"+transType+"_"+var1(i)+"_"+var2(j)+".mat");
                m4 = load("metrics/ISBI_"+subjects(4)+"_"+models(m)+"_"+transType+"_"+var1(i)+"_"+var2(j)+".mat");
                % Compute avg metrics 
                avg_ = struct;
                avg_.gt_metrics.dice = (m1.gt_metrics.dice + m2.gt_metrics.dice + m3.gt_metrics.dice + m4.gt_metrics.dice)/4;
                avg_.gt_metrics.sc =   (m1.gt_metrics.sc   + m2.gt_metrics.sc   + m3.gt_metrics.sc   + m4.gt_metrics.sc)/4;
                avg_.pred_metrics.dice = (m1.pred_metrics.dice + m2.pred_metrics.dice + m3.pred_metrics.dice + m4.pred_metrics.dice)/4;
                avg_.pred_metrics.sc =   (m1.pred_metrics.sc   + m2.pred_metrics.sc   + m3.pred_metrics.sc   + m4.pred_metrics.sc)/4;
                avg_.robust_metrics.dice = (m1.robust_metrics.dice + m2.robust_metrics.dice + m3.robust_metrics.dice + m4.robust_metrics.dice)/4;
                avg_.robust_metrics.sc =   (m1.robust_metrics.sc   + m2.robust_metrics.sc   + m3.robust_metrics.sc   + m4.robust_metrics.sc)/4;
                avg_.verTime = (m1.verTime + m2.verTime + m3.verTime + m4.verTime)/4;
                % Save metrics
                save("metrics/avg_ISBI_"+models(m)+"_"+transType+"_"+var1(i)+"_"+var2(j)+".mat","avg_");
            catch ME
                warning(ME.message)
            end
        end
    end
end