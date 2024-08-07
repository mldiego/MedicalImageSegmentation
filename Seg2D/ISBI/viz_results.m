%% Pull average metrics and create graphs for comparison

% Choose vars to go over
% var1 = [0.001, 0.002, 0.004]; %
% var2 = [1, 2, 5, 10]; % (nPix)
% transType = "linf";
%
% var1 = ["0.5"; "1"; "2"]; % (gamma)
% var2 = "0.0025";
% transType = "AdjustContrast";
%
% models = ["64", "80", "96"];

%% Load Data
% 64 - Linf
m64_0001_1  = load("metrics/avg_ISBI_64_linf_0.001_1.mat");
m64_0002_5  = load("metrics/avg_ISBI_64_linf_0.002_5.mat");
m64_0004_1  = load("metrics/avg_ISBI_64_linf_0.004_1.mat");
m64_0004_5  = load("metrics/avg_ISBI_64_linf_0.004_5.mat");
m64_0004_10 = load("metrics/avg_ISBI_64_linf_0.004_10.mat");
% 80 - Linf
m80_0001_1  = load("metrics/avg_ISBI_80_linf_0.001_1.mat");
m80_0002_5  = load("metrics/avg_ISBI_80_linf_0.002_5.mat");
m80_0004_1  = load("metrics/avg_ISBI_80_linf_0.004_1.mat");
m80_0004_5  = load("metrics/avg_ISBI_80_linf_0.004_5.mat");
m80_0004_10 = load("metrics/avg_ISBI_80_linf_0.004_10.mat");
% 96 - Linf
m96_0001_1  = load("metrics/avg_ISBI_96_linf_0.001_1.mat");
m96_0002_5  = load("metrics/avg_ISBI_96_linf_0.002_5.mat");
m96_0004_1  = load("metrics/avg_ISBI_96_linf_0.004_1.mat");
m96_0004_5  = load("metrics/avg_ISBI_96_linf_0.004_5.mat");
m96_0004_10 = load("metrics/avg_ISBI_96_linf_0.004_10.mat");
% 64 - AC
m64_05 = load("metrics/avg_ISBI_64_AdjustContrast_0.5_0.0025.mat");
m64_1  = load("metrics/avg_ISBI_64_AdjustContrast_1_0.0025.mat");
m64_2  = load("metrics/avg_ISBI_64_AdjustContrast_2_0.0025.mat");
% 80 - AC
m80_05 = load("metrics/avg_ISBI_80_AdjustContrast_0.5_0.0025.mat");
m80_1  = load("metrics/avg_ISBI_80_AdjustContrast_1_0.0025.mat");
m80_2  = load("metrics/avg_ISBI_80_AdjustContrast_2_0.0025.mat");

%% Visualizations (dice, sc, and time for ground truth metrics)
% 1) Linf

f = figure;
grid; hold on;
plot([1;2;3], [m64_0001_1.avg_.gt_metrics.dice; m64_0002_5.avg_.gt_metrics.dice;...
    m64_0004_10.avg_.gt_metrics.dice],'r', 'LineWidth', 2);
plot([1;2;3], [m80_0001_1.avg_.gt_metrics.dice; m80_0002_5.avg_.gt_metrics.dice;...
    m80_0004_10.avg_.gt_metrics.dice],'b', 'LineWidth', 2);
plot([1;2;3], [m96_0001_1.avg_.gt_metrics.dice; m96_0002_5.avg_.gt_metrics.dice;...
    m96_0004_10.avg_.gt_metrics.dice], '-', 'color', "#7E2F8E", 'LineWidth', 2);
% xlim([0.8 3.2]);
% ylim([0.57 0.61]);
xticks([1;2;3])
xticklabels({"[0.001, 1%]", "[0.002, 5%]", "[0.004, 10%]"});
xlabel('L_{\infty} variables [\epsilon, % pixels]');
ylabel('Dice')
ax = gca;
ax.FontSize = 16; 
legend({"64", "80", "96"}, 'Location','best');
% legend('boxoff');
exportgraphics(f, 'figures/linf_dice.pdf', 'ContentType', 'vector');


f = figure;
grid; hold on;
plot([1;2;3], [m64_0001_1.avg_.gt_metrics.sc; m64_0002_5.avg_.gt_metrics.sc;...
    m64_0004_10.avg_.gt_metrics.sc],'r', 'LineWidth', 2);
plot([1;2;3], [m80_0001_1.avg_.gt_metrics.sc; m80_0002_5.avg_.gt_metrics.sc;...
    m80_0004_10.avg_.gt_metrics.sc],'b', 'LineWidth', 2);
plot([1;2;3], [m96_0001_1.avg_.gt_metrics.sc; m96_0002_5.avg_.gt_metrics.sc;...
    m96_0004_10.avg_.gt_metrics.sc], '-', 'color', "#7E2F8E", 'LineWidth', 2);
% xlim([0.8 3.2]);
% ylim([0.57 0.61]);
xticks([1;2;3])
xticklabels({"[0.001, 1%]", "[0.002, 5%]", "[0.004, 10%]"});
xlabel('L_{\infty} variables [\epsilon, % pixels]');
ylabel('SC')
ax = gca;
ax.FontSize = 16; 
legend({"64", "80", "96"}, 'Location','best');
% legend('boxoff');
exportgraphics(f, 'figures/linf_sc.pdf', 'ContentType', 'vector');


f = figure;
grid; hold on;
plot([1;2;3], [m64_0001_1.avg_.verTime/900; m64_0002_5.avg_.verTime/900;...
    m64_0004_10.avg_.verTime/3600],'r', 'LineWidth', 2);
plot([1;2;3], [m80_0001_1.avg_.verTime/900; m80_0002_5.avg_.verTime/900;...
    m80_0004_10.avg_.verTime/3600],'b', 'LineWidth', 2);
plot([1;2;3], [m96_0001_1.avg_.verTime/900; m96_0002_5.avg_.verTime/900;...
    m96_0004_10.avg_.verTime/900], '-', 'color', "#7E2F8E", 'LineWidth', 2);
% xlim([0.8 3.2]);
% ylim([0.57 0.61]);
xticks([1;2;3])
xticklabels({"[0.001, 1%]", "[0.002, 5%]", "[0.004, 10%]"});
xlabel('L_{\infty} variables [\epsilon, % pixels]');
ylabel('V. Time (hours)')
ax = gca;
ax.FontSize = 16; 
legend({"64", "80", "96"}, 'Location','best');
% legend('boxoff');
exportgraphics(f, 'figures/linf_vt.pdf', 'ContentType', 'vector');



% 2) Adjust Contrast

f = figure;
grid; hold on;
plot([1;2;3], [m64_05.avg_.gt_metrics.dice; m64_1.avg_.gt_metrics.dice;...
    m64_2.avg_.gt_metrics.dice],'r', 'LineWidth', 2);
plot([1;2;3], [m80_05.avg_.gt_metrics.dice; m80_1.avg_.gt_metrics.dice;...
    m80_2.avg_.gt_metrics.dice],'b', 'LineWidth', 2);
% plot([1;2;3], [m96_05.avg_.gt_metrics.dice; m96_1.avg_.gt_metrics.dice;...
%     m96_2.avg_.gt_metrics.dice], '-', 'color', "#7E2F8E", 'LineWidth', 2);
% xlim([0.8 3.2]);
% ylim([0.57 0.61]);
xticks([1;2;3])
xticklabels({"0.5", "1", "2"});
xlabel('\gamma', 'fontweight','bold');
ylabel('Dice')
ax = gca;
ax.FontSize = 16; 
legend({"64", "80"}, 'Location','best');
% legend('boxoff');
exportgraphics(f, 'figures/ac_dice.pdf', 'ContentType', 'vector');

f = figure;
grid; hold on;
plot([1;2;3], [m64_05.avg_.gt_metrics.sc; m64_1.avg_.gt_metrics.sc;...
    m64_2.avg_.gt_metrics.sc],'r', 'LineWidth', 2);
plot([1;2;3], [m80_05.avg_.gt_metrics.dice; m80_1.avg_.gt_metrics.sc;...
    m80_2.avg_.gt_metrics.sc],'b', 'LineWidth', 2);
% plot([1;2;3], [m96_05.avg_.gt_metrics.dice; m96_1.avg_.gt_metrics.dice;...
%     m96_2.avg_.gt_metrics.dice], '-', 'color', "#7E2F8E", 'LineWidth', 2);
% xlim([0.8 3.2]);
% ylim([0.57 0.61]);
xticks([1;2;3])
xticklabels({"0.5", "1", "2"});
xlabel('\gamma', 'fontweight','bold');
ylabel('SC')
ax = gca;
ax.FontSize = 16; 
legend({"64", "80"}, 'Location','best');
% legend('boxoff');
exportgraphics(f, 'figures/ac_sc.pdf', 'ContentType', 'vector');


f = figure;
grid; hold on;
plot([1;2;3], [m64_05.avg_.verTime/900; m64_1.avg_.verTime/900;...
    m64_2.avg_.verTime/900],'r', 'LineWidth', 2);
plot([1;2;3], [m80_05.avg_.verTime/900; m80_1.avg_.verTime/900;...
    m80_2.avg_.verTime/900],'b', 'LineWidth', 2);
% plot([1;2;3], [m96_05.avg_.gt_metrics.dice; m96_1.avg_.gt_metrics.dice;...
%     m96_2.avg_.gt_metrics.dice], '-', 'color', "#7E2F8E", 'LineWidth', 2);
% xlim([0.8 3.2]);
% ylim([0.57 0.61]);
xticks([1;2;3])
xticklabels({"0.5", "1", "2"});
xlabel('\gamma', 'fontweight','bold');
ylabel('V. Time (hours)')
ax = gca;
ax.FontSize = 16; 
legend({"64", "80"}, 'Location','best');
% legend('boxoff');
exportgraphics(f, 'figures/ac_vt.pdf', 'ContentType', 'vector');