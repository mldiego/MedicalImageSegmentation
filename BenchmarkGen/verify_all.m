%% First, get the data loaded (same images used for all the models)

% Load data
rng(2023);
% Go through every folder (label) and load all 10 images per class
dataFolder = "../data/MedNIST";
categs = dir(dataFolder);
% Initialize vars
N = 120; % 20 images per class
XData = zeros(64, 64, 1, N); % (height, width, channels, batch)
YData = zeros(N, 1);
% Load images
count = 1;
for i = 3:length(categs)-1
    label = dir(dataFolder + "/"+ string(categs(i).name));
    indx_ = randperm(8500, 20);
    for k = indx_
        XData(:, :, :, count) = imread([label(k+2).folder '/' label(k+2).name]);
        YData(count) = i-2;
        count = count + 1;
    end
end

% 3) Adversarial attack (L_inf)
disturbance = [1,2]; 
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
        set_idx = (dd-1)*N+i;
        I(set_idx) = ImageStar(lb, ub);
    end
end
YData = repmat(YData, length(disturbance), 1);

%% Then verify all data with each model (5*3*3 = 45 models)

% Iterate trhough every folder and subfolder and analyze all the models
path = pwd;
folders = dir(path);
% Skip the first two that appear in every folder and subfolder as those
% correspond to (".", and "..")

% Go into every folder of and analyze each model
for r = 4:5 % iterate through regularizers (3)
    sub_path = [path, filesep, folders(r).name, filesep];
    inits_path = dir(sub_path);
    for i = 3:length(inits_path) % go through all initializations (3 x 3)
        if inits_path(i).isdir
            temp_path = [sub_path, inits_path(i).name, filesep, 'models', filesep];
            models_path = dir([temp_path, '*.mat']);
            for m = 1:length(models_path) % go through all models ( 5 x 3 x 3 )
                netpath = [temp_path, models_path(m).name];
                verify_medNist_model(netpath, I, YData);
            end
        end
    end
end

% 45 different files should be generated under the directory "BenchmarkGen/results/"

