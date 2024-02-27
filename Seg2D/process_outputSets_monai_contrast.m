%% Process results for all instances verified for one model

models = [64, 80, 96];
% sliceSize = [64, 80, 96, 112, 128];
% Any attacks on 112 and 128 run out of memory even on Windows computer
% RAM: 64GB
% AMD Ryzen 9 5900X 12-Core Processor, 3.70 GHz

% Consider only the 64 model with relax-star 0.95

for k = 1:length(models)

    sliceSize = models(k);

    % adversarial perturbation: linf
    results = dir("results/reach*AdjustContrast_"+string(sliceSize)+"*relax*.mat");
    data_path = "../../FMitF/Seg2D/data/matData/";
    
    N = length(results);
    
    % load model
    net = importONNXNetwork("models/size_"+string(sliceSize)+"/best_model.onnx");
    
    % compute riou and % rob for every reach set computed
    
    % do we want to verify anything else?
    
    for i = 1:N
        % get image id
        fileName = results(i).name;
        params = split(fileName,'_');
        imgID = params{5};
        
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
    
        % Load reachability data
        resData = load(['results/', fileName]);
    
        if isfield(resData, "ME") 
            if ~isempty(resData.ME)
                warning("There is no output set compute, an error was encountered.")
            end
        else
            
            % Get net's inference from img input
            yPred = predict(net, slice_img);
            yPred = (yPred > 0); % classify into 0 or 1
    
            % For faster verification, get bound estimates for each pixel
            [lb,ub] = resData.R.estimateRanges;
    
            %%% Compute results wrt to target output and predicted output
            
            % 1) get correctly classified as 0 (background)
            ver0 = (ub <= 0);
            ver0 = ~ver0;
            ver0pred = (ver0 == yPred) & (ver0 == 0);
            ver0label = (ver0 == slice_target) & (ver0 == 0);
            
            % 2) get correctly classified as 1 (lession)
            ver1 = (lb > 0);
            ver1pred = (ver1 == yPred) & (ver1 == 1);
            ver1label = (ver1 == slice_target) & (ver1 == 1);
    
            % 3) Get all correctly verified pixels
            VERIFIED_label = ver0label + ver1label; % quick estimate from bounds, we can look into the ones not verified
            VERIFIED_pred = ver0pred + ver1pred;  % quick estimate from bounds, we can look into the ones not verified
            
            % 4) Robustness value 
            rb_label = sum(VERIFIED_label, 'all')/numel(VERIFIED_label);
            rb_pred = sum(VERIFIED_pred, 'all')/numel(VERIFIED_pred);
    
            % 5) Pixels correctly classified in region of interest
            % only care about lession (label = 1 region)
            reg_ver_label = (VERIFIED_label == 1) & (slice_target == 1);
            rb_reg_label = sum(reg_ver_label, "all")/sum(slice_target, "all");
            reg_ver_pred = (VERIFIED_pred == 1) & (yPred == 1);
            rb_reg_pred = sum(reg_ver_pred, "all")/sum(yPred, "all"); % if not "interesting region" (aka 1s) are originally predicted, then it returns Nan
    
            % 6) IoU
            % a) get verified output
            verified_image = 2*ones(size(slice_img)); % pixel = 2 -> unknown
            background = find(ver0 == 0); % 0
            verified_image(background) = 0;
            lession = find(ver1 == 1); % 1
            verified_image(lession) = 1;
            % b) compute IoU score
            dice_label = jaccard(double(slice_target), double(verified_image));
            dice_label = dice_label(~isnan(dice_label));
            riou_label = sum(dice_label)/length(dice_label);
            dice_pred = dice(double(yPred), double(verified_image));
            dice_pred = dice_pred(~isnan(dice_pred));
            riou_pred = sum(dice_pred)/length(dice_pred);
    
            % 7) Save results
            params{1} = 'results';
            saveFile = join(params, "_");
            save(['results/', saveFile{1}], "dice_pred", "dice_label", "rb_reg_pred",...
                "rb_reg_label", "rb_pred", "rb_label");
        end
    
    end

end

