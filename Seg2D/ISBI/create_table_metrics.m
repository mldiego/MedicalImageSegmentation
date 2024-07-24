function create_table_metrics(subjectName, sliceSize, transType, var1, var2, pred_metrics, gt_metrics, robust_metrics, verTime)

    fileName = "metrics/ISBI_"+subjectName+"_"+sliceSize+"_"+transType+"_"+var1+"_"+var2+".tex";
    
    fileID = fopen(fileName,'w');
    fprintf(fileID, '\\begin{longtable}{ c  c | c | c  c  c  c  c  c  c c c}\n');
    if strcmp(transType, 'linf')
        fprintf(fileID, '\\toprule \\textbf{$\\epsilon$} & \\textbf{\\%% pixels} & & ');
    elseif contains(transType, "Adjust")
        fprintf(fileID, '\\toprule \\textbf{\\gamma} & \\textbf{$\\gamma_{range}$} & & ');
    else
        fprintf(fileID, '\\toprule \\textbf{coeff} & \\textbf{coeff$_{range}$} & & ');
    end
    fprintf(fileID, ['\\textbf{Dice} & \\textbf{PPV} & \\textbf{TPR} & \\textbf{LFPR} & ' ...
        '\\textbf{LTPR} & \\textbf{VD} & \\textbf{CORR} & \\textbf{SC} & \\textbf{V. Time} \\']);
    fprintf(fileID, '\\');
    fprintf(fileID, '\n');
    fprintf(fileID, '\\midrule \n');
    fprintf(fileID, '\\multirow{3}{*}{%s}  & \\multirow{3}{*}{%s} &', var1, var2);
    fprintf(fileID, '\\textbf{Validation} & '); % inference vs ground truth
    fprintf(fileID, '%.3f & ', pred_metrics.dice); 
    fprintf(fileID, '%.3f & ', pred_metrics.ppv);
    fprintf(fileID, '%.3f & ', pred_metrics.tpr);
    fprintf(fileID, '%.3f & ', pred_metrics.lfpr);
    fprintf(fileID, '%.3f & ', pred_metrics.ltpr);
    fprintf(fileID, '%.3f & ', pred_metrics.vd);
    fprintf(fileID, '%.3f & ', pred_metrics.corr);
    fprintf(fileID, '%.3f & ', pred_metrics.sc); 
    fprintf(fileID, '\\multirow{3}{*}{%.0f} \\', verTime);
    fprintf(fileID, '\\');
    fprintf(fileID, '\n');
    fprintf(fileID, ' & & \\textbf{Certified Scores} & '); % verification vs ground truth
    fprintf(fileID, '%.3f & ', gt_metrics.dice); 
    fprintf(fileID, '%.3f & ', gt_metrics.ppv);
    fprintf(fileID, '%.3f & ', gt_metrics.tpr);
    fprintf(fileID, '%.3f & ', gt_metrics.lfpr);
    fprintf(fileID, '%.3f & ', gt_metrics.ltpr);
    fprintf(fileID, '%.3f & ', gt_metrics.vd);
    fprintf(fileID, '%.3f & ', gt_metrics.corr);
    fprintf(fileID, '%.3f & \\', gt_metrics.sc);
    fprintf(fileID, '\\');
    fprintf(fileID, '\n');
    fprintf(fileID, '& & \\textbf{Robustness} & '); % true robustness: verification vs inference
    fprintf(fileID, '%.3f & ', robust_metrics.dice); 
    fprintf(fileID, '%.3f & ', robust_metrics.ppv);
    fprintf(fileID, '%.3f & ', robust_metrics.tpr);
    fprintf(fileID, '%.3f & ', robust_metrics.lfpr);
    fprintf(fileID, '%.3f & ', robust_metrics.ltpr);
    fprintf(fileID, '%.3f & ', robust_metrics.vd);
    fprintf(fileID, '%.3f & ', robust_metrics.corr);
    fprintf(fileID, '%.3f & \\', robust_metrics.sc);
    fprintf(fileID, '\\');
    fprintf(fileID, '\n');
    fprintf(fileID, '\\end{longtable}');
    fclose(fileID);

end

