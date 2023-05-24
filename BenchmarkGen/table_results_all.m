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
abdomen = [1:20, 121:140];
breast  = [21:40, 141:160];
chest   = [41:60, 161:180];
cxr     = [61:80, 181:200];
hand    = [81:100, 201:220];
head    = [101:120, 221:240];

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
idxs = 1:240; % all indexes for the rest of the results

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
idxs = 1:240; % all indexes for the rest of the results

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


%% Analysis per reg, per init (3 x 3)

% Order:
% dropout (glorot -> he -> narrow) -> jacobian (glorot -> he -> narrow) -> L2 (glorot -> he -> narrow)

regInitRes = zeros(Ninits*Nregs,4);
idxs = 1:240; % all indexes for the rest of the results

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

for i=1:N
    combo = floor(i/5);
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

% Figure comparing # robust
figure; % this figure does not look too good
plot(1:6, classRes(:,1),'r');
set(gca, 'xtick', 1:6)
set(gca, 'xticklabel', classNames)

% Figure comparing computation times
figure; % this figure does not look too good
plot(1:6, classRes(:,4),'b');
set(gca, 'xtick', 1:6)
set(gca, 'xticklabel', classNames)


%% Helper functions
% Process results per model
function [rob, unk, norob, avg_time] = process_model_res(res, idxs)
    rob   = sum(res(idxs,1) == 1);
    unk   = sum(res(idxs,1) == 2);
    norob = sum(res(idxs,1) == 0);
    avg_time = sum(res(idxs,2))/length(idxs);
end