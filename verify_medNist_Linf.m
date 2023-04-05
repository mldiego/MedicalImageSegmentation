%% Perform a robustness verification of the classification model trained on MedNIST
% model = simple CNN with 10 layers
%      accuracy = > 99%
% Perform adversarial attack (L infinity) on random images
% Robustness verification on these images

% 1) Load the model
model = load('model.mat');
net = matlab2nnv(model.net);
net.InputSize = net.Layers{1}.InputSize;
net.OutputSize = net.Layers{end-2}.OutputSize;

% 2) Load data
rng(2023);
% Go through every folder (label) and load all 10 images per class
dataFolder = "medNist/MedNIST";
categs = dir(dataFolder);
% Initialize vars
N = 60;
XData = zeros(64, 64, 1, N); % (height, width, channels, batch)
YData = zeros(N, 1);
% Load images
count = 1;
for i = 3:length(categs)-1
    label = dir(dataFolder + "/"+ string(categs(i).name));
    indx_ = randperm(8500, 10);
    for k = indx_
        XData(:, :, :, count) = imread([label(k+2).folder '/' label(k+2).name]);
        YData(count) = i-2;
        count = count + 1;
    end
end

% 3) Adversarial attack (L_inf)
disturbance = [1,2,3]; 
nD = length(disturbance);
ub_max = 255*ones(64,64);
lb_min = zeros(64,64);
I(N*nD) = ImageStar; % Initialize var for input sets
for dd = 1:nD
    epsilon = disturbance(dd);
    for i=1:N
        img = XData(:,:,:,i);
        lb = img - epsilon;
        lb = max(lb, lb_min); % ensure no negative values
        ub = img + epsilon;
        ub = min(ub, ub_max); % ensure no values > 255 (max pixel value)
        set_idx = (dd-1)*60+i;
        I(set_idx) = ImageStar(lb, ub);
    end
end
YData = repmat(YData, length(disturbance), 1);

% 4) Robustness Verification
reachOptions.reachMethod = 'approx-star';
res = zeros(N*nD,2);
for i=1:N*nD
    t = tic;
    res(i,1) = net.verify_robustness(I(i), reachOptions, YData(i));
    res(i,2) = toc(t);
end

% Save results 
save('medNist_results.mat', 'res');

fprintf('\n--------- RESULTS -----------\n');
disp("Robust = "+string(sum(res(:,1)==1))+ "/" + string(length(YData)));
disp("Unknown = "+string(sum(res(:,1)==2))+ "/" + string(length(YData)));
disp("Not robust = "+string(sum(res(:,1)==0))+ "/" + string(length(YData)));
disp("Computation time per image = "+string(mean(res(:,2))) + " seconds");
