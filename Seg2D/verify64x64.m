%% Questions
% 1) How are the slices cropped? 
% Original slices are 224x128, but input to NN is 128x128
%  - It is like a moving window all accross the image
% 
% We can rotate imrotate to rotate images, can specify method and angle to rotate each image with
% 2) What id the post processing done after inference on the model?
%  - Do rotation for image registration
%  - Focus on adversarial attacks or different brightening processes for
%  creating the input sets for segmentation


%% Load network
netMatlab = importONNXNetwork('models/size_64/best_model.onnx'); %model was saved with incorrect input size
% Create network in NNV
net = matlab2nnv(netMatlab);


%% Load data
data_path = "C:/Users/diego/Documents/Research/FMitF/Seg2D/data/matData/";
img_path = data_path + ".._data_axis_2_slice_101.mat";
% load single image
img = load(img_path);
target = img.mask;
img = img.flair;

img_size = size(img);
slice_size = 64;
target_size = [slice_size, slice_size];

r = centerCropWindow2d(img_size,target_size);
slice_img = imcrop(img, r);
slice_target = imcrop(target, r);

% slice1 = 129:end;
% slice2 = 1:96;

% slice_target = target(slice1, slice2);
% slice_img = img(slice1, slice2);

y = netMatlab.predict(slice_img);
y_mask = double(y > 0);

% Visualize images
figure;
subplot(2,2,1)
imshow(slice_img, [min(slice_img, [], 'all') max(slice_img, [], 'all')])
subplot(2,2,2)
imshow(slice_target, [min(slice_target, [], 'all') max(slice_target, [], 'all')])
subplot(2,2,3)
imshow(y, [min(y, [], 'all') max(y, [], 'all')])
subplot(2,2,4)
imshow(y_mask, [min(y_mask, [], 'all') max(y_mask, [], 'all')])


%% Reachability analysis

% Define reachability options
reachOptions = struct;
reachOptions.reachMethod = 'relax-star-range';
relaxFactors = [1, 0.9, 0.8];
% reachOptions.relaxFactor = 1;
epsilon = [0.0005; 0.001];
nPix = [10, 20, 30];

rT = zeros(length(relaxFactors), length(epsilon), length(nPix));
res = zeros(length(relaxFactors), length(epsilon), length(nPix));
errMsgs = {};

% Begin exploration
rng(0);
for rF = 1:length(relaxFactors)
    reachOptions.relaxFactor = relaxFactors(rF);
    for i = 1:length(epsilon)
        for j = 1:length(nPix)
            % Select random pixels to attack
            idxs = randperm(numel(slice_img),nPix(j));
            % Create bounds
            img_lb = slice_img;
            img_lb(idxs) = img_lb(idxs) - epsilon(i);
            img_ub = slice_img;
            img_ub(idxs) = img_ub(idxs) + epsilon(i);
            % Create input set
            IS = ImageStar(img_lb, img_ub);
            t = tic;
            try
                net.reach(IS, reachOptions);
                res(rF, i, j) = 1;
                disp("Well done");
            catch ME
                warning(ME.message);
                warning("Boooooooooooooooooooooooooooooooooooooo...");
                disp(sum(net.reachTime > 0));
                res(rF, i, j) = -1;
                errMsgs{rF,i,j} = ME;
            end
            rT(rF, i, j) = toc(t);
            disp("Finished with combo: (" + string(relaxFactors(rF)) + ", " + string(epsilon(i)) + ", "+string(nPix(j))+") is finished");
        end
    end
end

save('results/r64.mat', 'rT', 'res', "errMsgs");

