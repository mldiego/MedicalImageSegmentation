%% First, get the data loaded (same images used for all the models)

%% 1) Load data 

% Download if necessary
dataFolder = "../data/MedNIST";
if ~isfolder(dataFolder) % data is not downloaded yet
    mkdir("../data/");
    dataSource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz";
    gunzip(dataSource, 'medNist');
    untar('medNist/MedNIST.tar', "../data/");
end

% Go through every folder (label) and load all images
categs = dir(dataFolder);
% Initialize vars
Nall = 58954; % total number of images
XData = zeros(64, 64, 1, Nall); % height = width = 64, 10k imgs per class (greyscale)
YData = zeros(Nall, 1);        % except BreatMRI, that has only 8954
% Load images
count = 1;
for i = 3:length(categs)-1
    label = dir(dataFolder + "/"+ string(categs(i).name));
    for k = 3:length(label)
        XData(:, :, :, count) = imread([label(k).folder '/' label(k).name]);
        YData(count) = i-2;
        count = count + 1;
    end
end

% Get indxs to verify
verInfo = load("acc_results.mat");
XData = XData(:, :, :, verInfo.xVerIdxs);
YData = YData(verInfo.xVerIdxs);

N = length(YData); % total number of images to verify

% 3) Adversarial attack (L_inf)
disturbance = 3; 
nD = length(disturbance);
ub_max = 255*ones(64,64);
lb_min = zeros(64,64);
nR = 100;
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
        lb = reshape(lb, [64*64, 1]);
        ub = reshape(ub, [64*64, 1]);
        xB = Box(lb, ub); % lb, ub must be vectors
        xRand = xB.sample(nR-2);
        xRand = [lb, ub, xRand];
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
for r = 4:6 % iterate through regularizers (3)
    sub_path = [path, filesep, folders(r).name, filesep];
    inits_path = dir(sub_path);
    for i = 5:length(inits_path) % go through all initializations (3 x 3)
        if inits_path(i).isdir
            temp_path = [sub_path, inits_path(i).name, filesep, 'models', filesep];
            models_path = dir([temp_path, '*.mat']);
            for m = 1:length(models_path) % go through all models ( 5 x 3 x 3 )
                netpath = [temp_path, models_path(m).name];
                medNist_falsify_model(netpath, I, YData);
            end
        end
    end
end

% 45 different files should be generated under the directory "BenchmarkGen/results/"

