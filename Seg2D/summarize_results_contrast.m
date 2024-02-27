%% Process and summarize results from verification of semantic segmentation

% Load the .mat files
fileNames = dir("results/results*.mat");  % Replace with the actual file names
numFiles = numel(fileNames);

labelTable = zeros(3, 3, 2, 3, 10); % [# variables, # models, # epsilon, # pixels, # images]
predTable  = zeros(3, 3, 2, 3, 10); % [# variables, # models, # epsilon, # pixels, # images]
nansLabel   = zeros(3, 3, 2, 3, 10); % keep track of NaN results
nansPred    = zeros(3, 3, 2, 3, 10); % keep track of NaN results

% there should be 30 results file per model, but there are some with the
% larger epsilon and most number of pixels that did not finish
% how do we do with those ones? Skip it?

for i = 1:numFiles
    % Load the file
    data = load("results/" + fileNames(i).name);
    
    % Extract the field data
    x1 = data.rb_label;
    x2 = data.rb_pred;
    x3 = data.rb_reg_label;
    x4 = data.rb_reg_pred;
    x5 = data.riou_label;
    x6 = data.riou_pred;
    
    % Get nan values (e.g. when no 1s on the output)
    label = [x1;x3;x5];
    nanLabel = isnan(label);
    label(nanLabel) = 0;
    pred = [x2;x4;x6];
    nanPred = isnan(pred);
    pred(nanPred) = 0; % change all NaN to 0

    % get table indexes
    
    % a) Model
    if contains(fileNames(i).name, "linf_64")
        m = 1; % slice size 64
    elseif contains(fileNames(i).name, "linf_80")
        m = 2; % slice size 80
    else 
        m = 3; % slice size 96
    end
    
    % b) epsilon
    if endsWith(fileNames(i).name, '1.mat')
        j = 1; % 0.0001
    else
        j = 2; % 0.0005
    end
    
    % c) number of pixels
    if contains(fileNames(i).name, "_10_")
        k = 1; % 10 pixels modified
    elseif contains(fileNames(i).name, ["_409_0.", "_640_0.", "_921_0."])
        k = 2; % 10% of pixels
    else
        k = 3; % 20% of pixels
    end
    
    % d) image index
    if contains(fileNames(i).name, "_140_")
        id = 1; 
    elseif contains(fileNames(i).name, "_297_")
        id = 2; 
    elseif contains(fileNames(i).name, "_357_")
        id = 3; 
    elseif contains(fileNames(i).name, "_385_")
        id = 4;
    else % 414
        id = 5;
    end
    
    % Add scores to tables
    labelTable(:, m, j, k, id) = label; % [# variables, # models, # epsilon, # pixels, # images]
    predTable(:, m, j, k, id)  = pred;  % [# variables, # models, # epsilon, # pixels, # images]
    nansLabel(:, m, j, k, id)  = nanLabel;
    nansPred(:, m, j, k, id)   = nanPred;
    
end

% Compute score statistics (across imgs)
meanPred  = mean(predTable,   5, "omitnan");
stdPred   = std(predTable, 1, 5, "omitnan");
meanLabel = mean(labelTable,  5, "omitnan");
stdLabel = std(labelTable, 1, 5, "omitnan");

% Visualize results
% Results across attacks (epsilon + nPix), show all 3 models
% wrt prediction
plot_attack_results(meanPred, stdPred, 1, "robust_certifiedPixels_pred_all.pdf");    % 1 = rb_all
plot_attack_results(meanPred, stdPred, 2, "robust_certifiedPixels_pred_region.pdf"); % 2 = rb_reg
plot_attack_results(meanPred, stdPred, 3, "robust_iou_pred.pdf");                    % 3 = riou
% wrt label
plot_attack_results(meanLabel, stdLabel, 1, "robust_certifiedPixels_label_all.pdf");    % 1 = rb_all
plot_attack_results(meanLabel, stdLabel, 2, "robust_certifiedPixels_label_region.pdf"); % 2 = rb_reg
plot_attack_results(meanLabel, stdLabel, 3, "robust_iou_label.pdf");                    % 3 = riou


%% Helper functions
function plot_attack_results(meanVal, stdVal, score, figureName)

    modelNames = {"64"; "80"; "96"};
    colors = {'r'; 'b'; 'k'};
    x = [1; 2; 3; 4; 5];
    
    % Plotting the results
    fig = figure;
    
    for i = 1:3
        % get model values
        means = meanVal(score, i, :, :);
        means = reshape(means(1:5), [5 1]);
        stds = stdVal(score, i, :, :);
        stds = reshape(stds(1:5), [5 1]);

        %%% error bar plot
        % errorbar([1 2 3 4 5], means(1:end-1), stds(1:end-1), [colors{i},'o-'], 'LineWidth', 1.5);
        % hold on;

        %%% shaded area plot
        % Plot the shaded area
        f = fill([x; flip(x)]', [max(means - stds, 0); flip(min(means + stds,1))], colors{i});
        set(f, 'facealpha', 0.3); % Set transparency of the shaded area
        % the following line skip the name of the previous plot from the legend
        f.Annotation.LegendInformation.IconDisplayStyle = 'off';
        hold on;
        % Plot the mean line
        plot(x, means, 'o-', 'LineWidth', 1.5, 'Color', colors{i}, "DisplayName", modelNames{i});
        
    end
    grid;
    % ensure upper limit is 1
    yl = ylim;
    yl(2) = 1;
    ylim(yl);
    
    xticks(x);
    xticklabels({'a', 'b', 'c', 'd', 'e'});
    xlabel('Adversarial Perturbation');
    ylabel('Score');
    ax = gca;
    ax.FontSize = 14;
    % title('Comparison of Mean and Standard Deviation at Different Modified Pixels');
    legend('Location', 'best');
    exportgraphics(fig, "viz/" + figureName, 'ContentType', 'vector');

end

