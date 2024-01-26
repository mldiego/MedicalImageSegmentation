%% Save input and predictions for all sliced images used for vnnlib

rng(0);

% Study variables
sliceSizes = [64, 80, 96]; % for cropping and loading models
imgIdxs = randperm(315,5) + 100; % get 5 images from data

dataPath = '../../FMitF/Seg2D/data/matData/.._data_axis_2_slice_';% 101.mat';

%% Begin creating vnnlib files for all img (adversarial combos)

% Begin exploration
for i=1:length(sliceSizes)
    
    sZ = sliceSizes(i); % to choose models and reshape data into
    % Load model
    net = importONNXNetwork("models/size_"+string(sZ)+"/best_model.onnx");
            
    for m = 1:length(imgIdxs)

        idx = imgIdxs(m);

        infer_and_save(net, dataPath, sZ, idx);

    end

end


%% Helper functions

function infer_and_save(net, dataPath, sliceSize, imgIdx)

    rng(2024); % to select the same pixels for all models

    % create file name
    name = "sliceData/img_"+string(imgIdx)+"_sliceSize_"+string(sliceSize);
    outfile = name+".mat";

    % Load data
    data = load([dataPath num2str(imgIdx) '.mat']);
    img = data.flair; % input
    target = data.mask; % output (target)
    img_size = size(img); % size of I/O
    
    % Preprocess data
    target_size = [sliceSize sliceSize];
    r = centerCropWindow2d(img_size, target_size); % cropping window
    slice_img = imcrop(img, r); % sliced input image
    slice_target = imcrop(target, r); % sliced target output

    % Compute model output
    yPred = predict(net, slice_img);
    yPred = (yPred > 0); % classify into 0 or 1
    yPred = single(yPred);

    %%% Compute results

    % Robustness value 
    rb = sum(yPred == slice_target, 'all')/numel(yPred);

    % IoU (jaccard score)
    iou = jaccard(double(yPred), double(slice_target));
    iou = iou(~isnan(iou));
    
    % Dice score
    diceScore = dice(double(yPred), double(slice_target));
    diceScore = diceScore(~isnan(diceScore));
    
    % Save results
    save(outfile, "slice_img", "slice_target", "yPred", "rb", "iou", "diceScore");

end



