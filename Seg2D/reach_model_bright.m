function [R,rT] = reach_model_bright(sliceSize, imgIdx, reachOptions, attack)
   
    %% Load network
    
    netMatlab = importONNXNetwork("models/size_"+string(sliceSize)+"/best_model.onnx");
    net = matlab2nnv(netMatlab);
    
    %% Load data
    
    % load single data point
    data_path = "../../FMitF/Seg2D/data/matData/";
    img_path = data_path + ".._data_axis_2_slice_"+string(imgIdx)+".mat";
    data = load(img_path);
    img = data.flair;
    
    img_size = size(img);
    target_size = [sliceSize, sliceSize];
    r = centerCropWindow2d(img_size, target_size);
    slice_img = imcrop(img, r);

    %% Get attack data
    
    nPix = attack.nPix;
    epsilon = attack.epsilon;
    threshold = attack.threshold;
    
    % Select random pixels to attack
    rng(2024); % to select the same pixels for all models
    if isa(nPix, 'string') % strings define percentage of pixels in image
        nPix = floor(sliceSize^2 * str2double(nPix)/100);
    end
        
    %% Define input set
    
    IS = bright_attack(slice_img, nPix, threshold, epsilon);

    %% Compute reach sets
    
    t = tic;
    ME = [];
    try
        R = net.reach(IS, reachOptions);
    catch ME
        R = net.reachSet;
    end
    rT = toc(t);

    save("results/reach_bright_"+string(sliceSize)+"_"+string(imgIdx)+"_"+string(nPix)+"_"+string(epsilon)+".mat", "R", "rT", "ME", "-v7.3");


end

% Return a ImageStar of a brightening attack on a few pixels
function I = bright_attack(im, max_pixels, threshold, noise_disturbance)

    % Initialize vars
    ct = 0; % keep track of pixels modified
    flag = 0; % determine when to stop modifying pixels
    im = single(im);
    at_im = im;

    % Create brightening attack
    for i=1:size(im,1)
        for j=1:size(im,2)
            if im(i,j) < threshold
                at_im(i,j) = 255;
                ct = ct + 1;
                if ct >= max_pixels
                    flag = 1;
                    break;
                end
            end
            if flag == 1
                break
            end
        end
        if flag == 1
            break;
        end
    end

    % Define input set as VolumeStar
    dif_vol = -im + at_im;
    noise = dif_vol;
    V(:,:,:,1) = im;   % center of set
    V(:,:,:,2) = noise; % basis vectors
    C = [1; -1];          % constraints
    d = [1-noise_disturbance; -1]; % constraints
    I = ImageStar(V, C, d, 1-noise_disturbance, 1); % input set

end

