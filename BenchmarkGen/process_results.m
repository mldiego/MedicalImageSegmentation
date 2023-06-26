%% Process results
results_folder = dir('results');

% N = 45;

% first load all results
N = length(results_folder)-2;
allRes = cell(N,2);

for i=3:N+2
    ind_res = load(['results', filesep, results_folder(i).name]);
    allRes{i-2,1} = ind_res.res;
    allRes{i-2,2} = results_folder(i).name;
end

%% Analysis per class

% res indexes per class
abdomen = 1:50;
breast  = 51:100;
chest   = 101:150;
cxr     = 151:200;
hand    = 201:250;
head    = 251:300;

classes = [abdomen; breast; chest; cxr; hand; head];
numClasses = 6;
classRes = zeros(numClasses,4);

% Process results per class
for c = 1:numClasses
    idxs = classes(c,:);
    Rob = 0; Unk = 0; Norob = 0; Avgtime = 0;
    for i=1:N
        [rob, unk, norob, time] = process_model_res(allRes{i,1}, idxs);
        Rob = Rob + rob; Unk = Unk + unk; 
        Norob = Norob + norob; Avgtime = Avgtime + time;
    end
    Avgtime = Avgtime/N;
    classRes(c,:) = [Rob, Unk, Norob, Avgtime];
end

%% Analysis per regularizer

regs = ["dropout", "jacobian", "l2"];
Nregs = length(regs);
regRes = zeros(Nregs,4); % dropout, jacobian, l2
idxs = 1:300; % all indexes for the rest of the results

for r = 1:Nregs
    Rob = 0; Unk = 0; Norob = 0; Avgtime = 0;
    count = 0;
    for i=1:N
        if contains(allRes{i,2}, regs(r))
            [rob, unk, norob, time] = process_model_res(allRes{i,1}, idxs);
            Rob = Rob + rob; Unk = Unk + unk; 
            Norob = Norob + norob; Avgtime = Avgtime + time;
            count = count + 1;
        end
    end
    Avgtime = Avgtime/count;
    regRes(r,:) = [Rob, Unk, Norob, Avgtime];
end

%% Analysis per initialization scheme

inits = ["glorot", "he", "narrow"]; 
Ninits = length(inits);
initRes = zeros(Ninits,4);
idxs = 1:300; % all indexes for the rest of the results

for r = 1:Ninits
    Rob = 0; Unk = 0; Norob = 0; Avgtime = 0;
    count = 0;
    for i=1:N
        if contains(allRes{i,2}, inits(r))
            [rob, unk, norob, time] = process_model_res(allRes{i,1}, idxs);
            Rob = Rob + rob; Unk = Unk + unk; 
            Norob = Norob + norob; Avgtime = Avgtime + time;
            count = count + 1;
        end
    end
    Avgtime = Avgtime/count;
    initRes(r,:) = [Rob, Unk, Norob, Avgtime];
end

%% Analysis per seed (0,1,2,3,4)

seeds = ["0", "1", "2", "3", "4"]; 
Nseeds = length(seeds);
seedRes = zeros(Nseeds,4);
idxs = 1:300; % all indexes for the rest of the results

for i=1:N
    loc_seed = mod(i,5);
    if loc_seed == 0
        loc_seed = 5;
    end
    [rob, unk, norob, time] = process_model_res(allRes{i,1}, idxs);
    seedRes(loc_seed, 1) = seedRes(loc_seed, 1)  + rob/2700; % 9 (3 init x 3 reg) * 300 (images) = 2700
    seedRes(loc_seed, 2) = seedRes(loc_seed, 2)  + unk/2700;
    seedRes(loc_seed, 3) = seedRes(loc_seed, 3)  + norob/2700;
    seedRes(loc_seed, 4) = seedRes(loc_seed, 4)  + time/9;
end


%% Analysis per reg, per init (3 x 3)

% Order:
% dropout (glorot -> he -> narrow) -> jacobian (glorot -> he -> narrow) -> L2 (glorot -> he -> narrow)

regInitRes = zeros(Ninits*Nregs,4);
idxs = 1:300; % all indexes for the rest of the results

for i=1:N
    combo = ceil(i/5);
    [rob, unk, norob, time] = process_model_res(allRes{i,1}, idxs);
    regInitRes(combo,1) = regInitRes(combo,1) + rob; % robust
    regInitRes(combo,2) = regInitRes(combo,2) + unk; % unknown
    regInitRes(combo,3) = regInitRes(combo,3) + norob; % not robust
    regInitRes(combo,4) = regInitRes(combo,4) + time; % computation time
end
regInitRes(:,4) = regInitRes(:,4)/5; % 5 = number of models per combo


%% Analysis per reg per class

% Order:
% dropout (1,2,3,4,5,6) -> jacobian (1,2,3,4,5,6) -> L2 (1,2,3,4,5,6)

regClassRes = zeros(numClasses*Nregs,4);

% Process results per class
for c = 1:numClasses
    idxs = classes(c,:);
    for i=1:N
        [rob, unk, norob, time] = process_model_res(allRes{i,1}, idxs);
        if contains(allRes{i,2}, "dropout")
            regClassRes(c,1) = regClassRes(c,1) + rob;
            regClassRes(c,2) = regClassRes(c,2) + unk;
            regClassRes(c,3) = regClassRes(c,3) + norob;
            regClassRes(c,4) = regClassRes(c,4) + time;
        elseif contains(allRes{i,2}, "jacobian")
            regClassRes(c+6,1) = regClassRes(c+6,1) + rob;
            regClassRes(c+6,2) = regClassRes(c+6,2) + unk;
            regClassRes(c+6,3) = regClassRes(c+6,3) + norob;
            regClassRes(c+6,4) = regClassRes(c+6,4) + time;
        else % L2
            regClassRes(c+12,1) = regClassRes(c+12,1) + rob;
            regClassRes(c+12,2) = regClassRes(c+12,2) + unk;
            regClassRes(c+12,3) = regClassRes(c+12,3) + norob;
            regClassRes(c+12,4) = regClassRes(c+12,4) + time;
        end
    end
end
regClassRes(:,4) = regClassRes(:,4)/15; % (5 models per init, so 15 models total)


%% Analysis per reg per init per class

regInitClassRes = zeros(numClasses*Nregs*Ninits,4);
% dropout (1 to 18)
% jacobian (19 to 36)
% l2 (37 to 54)

for i=1:N
    combo = floor((i-1)/5);
    for c = 1:numClasses
        idxs = classes(c,:);
        [rob, unk, norob, time] = process_model_res(allRes{i,1}, idxs);
        regInitClassRes(combo*6+c,1) = regInitClassRes(combo*6+c,1) + rob; % robust
        regInitClassRes(combo*6+c,2) = regInitClassRes(combo*6+c,2) + unk; % unknown
        regInitClassRes(combo*6+c,3) = regInitClassRes(combo*6+c,3) + norob; % not robust
        regInitClassRes(combo*6+c,4) = regInitClassRes(combo*6+c,4) + time; % computation time
    end
end
regInitClassRes(:,4) = regInitClassRes(:,4)/5; % 5 = number of models per combo

classNames = ["AbdomenCT", "BreatMRI", "ChestCT", "CXR", "Hand", "HeadCT"];


%% Visualize results


%% Compare regularization vs classes

dropout_class = regInitClassRes(1:6,:) + regInitClassRes(7:12,:) + regInitClassRes(13:18,:);
dropout_class(:,4) = dropout_class(:,4)/3; % compute average time
dropout_class(:,1:3) = dropout_class(:,1:3)/900; % total number of instances (plot as percentage)
jacobian_class = regInitClassRes(19:24,:) + regInitClassRes(25:30,:) + regInitClassRes(31:36,:);
jacobian_class(:,4) = jacobian_class(:,4)/3;
jacobian_class(:,1:3) = jacobian_class(:,1:3)/900; % total number of instances (plot as percentage)
l2_class= regInitClassRes(37:42,:) + regInitClassRes(43:48,:) + regInitClassRes(49:54,:);
l2_class(:,4) = l2_class(:,4)/3;
l2_class(:,1:3) = l2_class(:,1:3)/900; % total number of instances (plot as percentage)

% Create figure
figure;
grid;hold on;
plot(1:6, dropout_class(:,1),'r-d');
plot(1:6, jacobian_class(:,1),'b-o');
plot(1:6, l2_class(:,1),'k--');
set(gca, 'xtick', 1:6)
set(gca, 'xticklabel', classNames);
% set(gca, "YTick", 0.9:0.01:1);
% ylim([0.925, 1.005])
ylabel("Robust %");
legend('dropout','jacobian', 'L2', 'Location','best');
exportgraphics(gca, "plots/regRes_vs_class.pdf",'ContentType','vector');

% Create figure
figure;
grid;hold on;
plot(1:6, dropout_class(:,4),'r-d');
plot(1:6, jacobian_class(:,4),'b-o');
plot(1:6, l2_class(:,4),'k--');
set(gca, 'xtick', 1:6)
set(gca, 'xticklabel', classNames);
ylabel("Time (s)")
legend('dropout','jacobian', 'L2', 'Location','best');
exportgraphics(gca, "plots/regTime_vs_class.pdf",'ContentType','vector');


%% Compare initializers vs classes 

glorot_class = regInitClassRes(1:6,:) + regInitClassRes(19:24,:) + regInitClassRes(37:42,:);
glorot_class(:,4) = glorot_class(:,4)/15; % compute average time
glorot_class(:,1:3) = glorot_class(:,1:3)/900; % total number of instanes (plot as percentage)
he_class = regInitClassRes(7:12,:) + regInitClassRes(25:30,:) + regInitClassRes(43:48,:) ;
he_class(:,4) = he_class(:,4)/15;
he_class(:,1:3) = he_class(:,1:3)/900; % total number of instanes (plot as percentage)
narrow_class = regInitClassRes(13:18,:) + regInitClassRes(31:36,:) + regInitClassRes(49:54,:);
narrow_class(:,4) = narrow_class(:,4)/15;
narrow_class(:,1:3) = narrow_class(:,1:3)/900; % total number of instanes (plot as percentage)

% Create figure
figure;
grid;hold on;
plot(1:6, glorot_class(:,1),'r-d');
plot(1:6, he_class(:,1),'b-o');
plot(1:6, narrow_class(:,1),'k--');
set(gca, 'xtick', 1:6)
set(gca, 'xticklabel', classNames);
% set(gca, "YTick", 0.9:0.01:1);
% ylim([0.925, 1.005])
ylabel("Robust %");
legend('glorot','he', 'narrow-normal', 'Location','best');
exportgraphics(gca, "plots/initRes_vs_class.pdf",'ContentType','vector');

% Create figure
figure;
grid;hold on;
plot(1:6, glorot_class(:,4),'r-d');
plot(1:6, he_class(:,4),'b-o');
plot(1:6, narrow_class(:,4),'k--');
set(gca, 'xtick', 1:6)
set(gca, 'xticklabel', classNames);
ylabel("Time (s)")
legend('glorot','he', 'narrow-normal', 'Location','best');
exportgraphics(gca, "plots/initTime_vs_class.pdf",'ContentType','vector');

%% Compare combinations vs classes 

% Create figure
figure;
grid;hold on;
plot(1:6, regInitClassRes(1:6,1)/250,'--d', 'Color', "#A2142F");
plot(1:6, regInitClassRes(7:12,1)/250,'b-o');
plot(1:6, regInitClassRes(13:18,1)/250,'k--.');
plot(1:6, regInitClassRes(19:24,1)/250,'m-x');
plot(1:6, regInitClassRes(25:30,1)/250,'-v', 'Color', "#EDB120");
plot(1:6, regInitClassRes(31:36,1)/250,'--', 'Color','#808080');
plot(1:6, regInitClassRes(37:42,1)/250,'->', 'Color', "#D95319");
plot(1:6, regInitClassRes(43:48,1)/250,'-s', 'Color', "#7E2F8E");
plot(1:6, regInitClassRes(49:54,1)/250,'--+', 'Color', "#77AC30");
set(gca, 'xtick', 1:6)
set(gca, 'xticklabel', classNames);
% set(gca, "YTick", 0.9:0.01:1);
% ylim([0.925, 1.005])
ylabel("Robust %");
legend('dropout_G','dropout_H', 'dropout_N', 'jacobian_G','jacobian_H',...
    'jacobian_N', 'L2_G','L2_H', 'L2_N', 'Location','best');
exportgraphics(gca, "plots/comboRes_vs_class.pdf",'ContentType','vector');

% Create figure
figure;
grid; hold on;
plot(1:6, regInitClassRes(1:6,4),'--d', 'Color', "#A2142F");
plot(1:6, regInitClassRes(7:12,4),'b-o');
plot(1:6, regInitClassRes(13:18,4),'k--.');
plot(1:6, regInitClassRes(19:24,4),'m-x');
plot(1:6, regInitClassRes(25:30,4),'-v', 'Color', "#EDB120");
plot(1:6, regInitClassRes(31:36,4),'--', 'Color','#808080');
plot(1:6, regInitClassRes(37:42,4),'->', 'Color', "#D95319");
plot(1:6, regInitClassRes(43:48,4),'-s', 'Color', "#7E2F8E");
plot(1:6, regInitClassRes(49:54,4),'--+', 'Color', "#77AC30");
set(gca, 'xtick', 1:6)
set(gca, 'xticklabel', classNames);
ylabel("Time (s)")
legend('dropout_G','dropout_H', 'dropout_N', 'jacobian_G','jacobian_H',...
    'jacobian_N', 'L2_G','L2_H', 'L2_N', 'Location','best');
exportgraphics(gca, "plots/comboTime_vs_class.pdf",'ContentType','vector');

%% Helper functions

% Process results per model
function [rob, unk, norob, avg_time] = process_model_res(res, idxs)
    rob   = sum(res(idxs,1) == 1);
    unk   = sum(res(idxs,1) == 2);
    norob = sum(res(idxs,1) == 0);
    avg_time = sum(res(idxs,2))/length(idxs);
end

