function create_table_metrics(subjectName, sliceSize, transType, var1, var2, pred_metrics, gt_metrics, robust_metrics, verTime)

    fileName = "metrics/UMCL_"+subjectName+"_"+sliceSize+"_"+transType+"_"+var1+"_"+var2+".tex";
    
    fileID = fopen(fileName,'w');
    fprintf(fileID, '\begin{longtable}{ c  c | c | c  c  c  c  c  c  c c c}\n');
    fprintf(fileID, '\toprule\textbf{\# pixels} & \textbf{$\epsilon$} & & \textbf{Dice} & \textbf{PPV} & \textbf{TPR} & \textbf{LFPR} & \textbf{LTPR} & \textbf{VD} & \textbf{CORR} & \textbf{SC} & \textbf{V. Time} \\');
    fprintf(fileID, '\n');
    fprintf(fileID, '\midrule \n');
    fprintf(fileID, '\multirow{3}{*}{%s}  & \multirow{3}{*}{%s} &', var1, var2);
    fprintf(fileID, '\textbf{Validation} & '); % inference vs ground truth
    fprintf(fileID, '%f &', pred_metrics.dice); 
    fprintf(fileID, '%f &', pred_metrics.ppv);
    fprintf(fileID, '%f &', pred_metrics.tpr);
    fprintf(fileID, '%f &', pred_metrics.lfpr);
    fprintf(fileID, '%f &', pred_metrics.ltpr);
    fprintf(fileID, '%f &', pred_metrics.vd);
    fprintf(fileID, '%f &', pred_metrics.corr);
    fprintf(fileID, '%f &', pred_metrics.sc); 
    fprintf(fileID, '\multirow{3}{*}{%f} \\', verTime);
    fprintf(fileID, '\n');
    fprintf(fileID, '\textbf{Certified Scores} & '); % verification vs ground truth
    fprintf(fileID, '%f &', gt_metrics.dice); 
    fprintf(fileID, '%f &', gt_metrics.ppv);
    fprintf(fileID, '%f &', gt_metrics.tpr);
    fprintf(fileID, '%f &', gt_metrics.lfpr);
    fprintf(fileID, '%f &', gt_metrics.ltpr);
    fprintf(fileID, '%f &', gt_metrics.vd);
    fprintf(fileID, '%f &', gt_metrics.corr);
    fprintf(fileID, '%f & \\', gt_metrics.sc); 
    fprintf(fileID, '\n');
    fprintf(fileID, '\textbf{Robustness} & '); % true robustness: verification vs inference
    fprintf(fileID, '%f &', robust_metrics.dice); 
    fprintf(fileID, '%f &', robust_metrics.ppv);
    fprintf(fileID, '%f &', robust_metrics.tpr);
    fprintf(fileID, '%f &', robust_metrics.lfpr);
    fprintf(fileID, '%f &', robust_metrics.ltpr);
    fprintf(fileID, '%f &', robust_metrics.vd);
    fprintf(fileID, '%f &', robust_metrics.corr);
    fprintf(fileID, '%f & \\', robust_metrics.sc); 
    fprintf(fileID, '\n');
    fclose(fileID);

end

