%% Create long table with all resutls for each perturbation

% Let's create one table per model per perturbation
models = ["64", "80", "96"];

%% Adjust Contrast

for k = 1:length(models)
    sZ = models(k);

    results = dir("results/results_monai_AdjustContrast_"+sZ+"*.mat");

    % [gamma, gamma_range, img] as part of the input transform variables
    % metrics to look into: [vTime, acc, L-acc, dice, ppv, tpr, fpr] 
    
    % begin table
    fid = fopen("tableSummary_AdjustContrast"+sZ+".tex", "w");
    % fprintf(fid, "\\scriptsize\n");
    fprintf(fid, "\\begin{longtable}{ c  c  c | c | c  c  c  c  c  c  c }\n");
    fprintf(fid, "\\toprule");
    fprintf(fid, "\\textbf{$\\gamma$} & \\textbf{$\\epsilon$} & \\textbf{img} & & \\textbf{acc} & \\textbf{L-acc} & \\textbf{DSC} & \\textbf{PPV} & \\textbf{TPR} & \\textbf{FPR} & \\textbf{V. Time} \\");
    fprintf(fid, "\\");
    fprintf(fid, "\n");
    fprintf(fid, "\\midrule\n");
    % process all results
    for i=1:height(results)
        name = results(i).name;
        res = load("results/"+name);
        % reduce length of variables
        iR = res.inference_metrics;
        gtR = res.verifiedGT_metrics;
        pR = res.verifiedPred_metrics;
        % get transform variables
        name = split(name, '_');
        img = name{5};
        gamma = name{6};
        gRange = name{7};
        % create table line
        fprintf(fid, "\\multirow{3}{*}{%s} & \\multirow{3}{*}{%s} & \\multirow{3}{*}{%s} & ", gamma, gRange, img); % perturbation specs
        fprintf(fid, " \\textbf{Inference} & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f ", iR.cra, iR.lcra, iR.dsc, iR.ppv, iR.tpr,iR.fpr); % metrics (inference)
        fprintf(fid, " & \\multirow{3}{*}{%.3f} \\", res.rT); % reachability time
        fprintf(fid, "\\");
        fprintf(fid, "\n");
        fprintf(fid, " & & & \\textbf{Ground Truth} & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & \\", gtR.cra, gtR.lcra, gtR.dsc, gtR.ppv, gtR.tpr,gtR.fpr); % metrics (ground truth)
        fprintf(fid, "\\");
        fprintf(fid, "\n");
        fprintf(fid, " & & & \\textbf{Pred. Label} & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & \\", pR.cra, pR.lcra, pR.dsc, pR.ppv, pR.tpr, pR.fpr); % metrics (prediction)
        fprintf(fid, "\\");
        fprintf(fid, " \\hline ");
        fprintf(fid, "\n");
    end
    fprintf(fid, "\\bottomrule\n"); 
    fprintf(fid, "\\end{longtable}\n");
    % fprintf(fid, "\\end{center}");
    fclose(fid);

end



%% Bias Field

for k = 1:length(models)

    sZ = models(k);

    results = dir("results/results_monai_BiasField_"+sZ+"*.mat");

    % [cefficient, coeff_range, img] as part of the input transform variables (order is always 3)
    % metrics to look into: [vTime, acc, L-acc, dice, ppv, tpr, fpr] 
    
    % begin table
    fid = fopen("tableSummary_BiasField"+sZ+".tex", "w");
    % fprintf(fid, "\\scriptsize\n");
    fprintf(fid, "\\begin{longtable}{ c  c  c | c | c  c  c  c  c  c  c }\n");
    fprintf(fid, "\\toprule");
    fprintf(fid, "\\textbf{Coeff} & \\textbf{$\\epsilon$} & \\textbf{img} & & \\textbf{acc} & \\textbf{L-acc} & \\textbf{DSC} & \\textbf{PPV} & \\textbf{TPR} & \\textbf{FPR} & \\textbf{V. Time} \\");
    fprintf(fid, "\\");
    fprintf(fid, "\n");
    fprintf(fid, "\\midrule\n");
    % process all results
    for i=1:height(results)
        name = results(i).name;
        res = load("results/"+name);
        % reduce length of variables
        iR = res.inference_metrics;
        gtR = res.verifiedGT_metrics;
        pR = res.verifiedPred_metrics;
        % get transform variables
        name = split(name, '_');
        img = name{5};
        coeff = name{7};
        cRange = name{8};
        % create table line
        fprintf(fid, "\\multirow{3}{*}{%s} & \\multirow{3}{*}{%s} & \\multirow{3}{*}{%s} & ", coeff, cRange, img); % perturbation specs
        fprintf(fid, " \\textbf{Inference} & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f ", iR.cra, iR.lcra, iR.dsc, iR.ppv, iR.tpr,iR.fpr); % metrics (inference)
        fprintf(fid, " & \\multirow{3}{*}{%.3f} \\", res.rT); % reachability time
        fprintf(fid, "\\");
        fprintf(fid, "\n");
        fprintf(fid, " & & & \\textbf{Ground Truth} & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & \\", gtR.cra, gtR.lcra, gtR.dsc, gtR.ppv, gtR.tpr,gtR.fpr); % metrics (ground truth)
        fprintf(fid, "\\");
        fprintf(fid, "\n");
        fprintf(fid, " & & & \\textbf{Pred. Label} & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & \\", pR.cra, pR.lcra, pR.dsc, pR.ppv, pR.tpr, pR.fpr); % metrics (prediction)
        fprintf(fid, "\\");
        fprintf(fid, " \\hline ");
        fprintf(fid, "\n");
    end
    fprintf(fid, "\\bottomrule\n"); 
    fprintf(fid,"\\end{longtable}\n");
    % fprintf(fid,"\\end{center}");
    fclose(fid);

end
