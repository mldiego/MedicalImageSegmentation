% Compute metrics/results for every reachable set computed from monai
% transformations and other perturbations

models = [64, 80, 96];
% sliceSize = [64, 80, 96, 112, 128];
% Any attacks on 112 and 128 run out of memory even on Windows computer
% Windows Specs
%   RAM: 64GB
%   AMD Ryzen 9 5900X 12-Core Processor, 3.70 GHz
% Linux Specs (where ended up running things from bash scripts)
%   RAM: 64 GB
%   Intel® Core™ i7-7700 CPU @ 3.60GHz × 8 

for k = 1:length(models)

    sliceSize = models(k);

    % adversarial perturbation: any monai perturbation
    results = dir("results/reach_monai_*"+"_"+string(sliceSize)+"_"+"*.mat");
    data_path = "../../FMitF/Seg2D/data/matData/";
    
    N = length(results);
    
    % load model
    net = importONNXNetwork("models/size_"+string(sliceSize)+"/best_model.onnx");
        
    % do we want to verify anything else?
    
    for i = 1:N
        
        % get image id
        fileName = results(i).name;
        params = split(fileName,'_');
        imgID = params{5}; % 4 for non-monai transforms
        
        % load single data point
        img_path = data_path + ".._data_axis_2_slice_"+string(imgID)+".mat";
        data = load(img_path);
        img = data.flair;
        target = data.mask;
        
        % Process img (i/o)
        img_size = size(target);
        target_size = [sliceSize, sliceSize];
        r = centerCropWindow2d(img_size, target_size);
        slice_target = imcrop(target, r);
        slice_img = imcrop(img, r);
        yPred = predict(net, slice_img);
        
        % Compute all metrics
        [inference_metrics, verifiedGT_metrics, verifiedPred_metrics, rT] = semantic_segmentation_metrics(slice_target, yPred, ['results/', fileName]);
        
        % 7) Save results
        params{1} = 'results';
        saveFile = join(params, "_");
        save(['results/', saveFile{1}], "inference_metrics", "verifiedPred_metrics", "verifiedGT_metrics", "rT");
    
    end

end