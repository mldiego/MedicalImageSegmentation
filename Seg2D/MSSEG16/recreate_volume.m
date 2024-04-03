function [ver_vol, pred_vol, verTime] = recreate_volume(net, flair, sliceSize, sbName, transType, v1, v2)
    % get all results files
    resFiles = dir("results/reach_monai_"+transType+"_"+string(sliceSize)+"_"+sbName+"*_"+string(v1)+"_"+string(v2)+"_*.mat");
    resFiles = struct2cell(resFiles);
    resFiles = resFiles(1,:);

    pred_vol = zeros(size(flair));
    ver_vol = zeros(size(flair));
    verTime = 0;
    
    % Go trough every slice in the data
    for c=1:size(flair,1)
        [pred_vol(c,:,:), ver_vol(c,:,:), verTime] = get_patch_data(net, flair, sliceSize, c, sbName, transType, v1, v2, resFiles, verTime);
    end
    

end

% Generate starting points for slices
function [pred_c, ver_c, verTime] = get_patch_data(net, flair, sZ, c, sbName, transType, epsilon, nPix, resFiles, verTime)

    xC = 1:sZ:size(flair,2); % initialize window coordinates
    if xC(end) > size(flair,2)
        xC = xC(1:end-1);
    end
    yC = 1:sZ:size(flair,3);
    if xC(end) > size(flair,3)
        xC = xC(1:end-1);
    end

    flair_slice   = squeeze(flair(c,:,:));

    width = size(flair_slice,2);
    height = size(flair_slice,1);

    pred_c = zeros(height,width);
    ver_c = zeros(height, width);

    for i = xC
        for j = yC

            % Get data info
            % flair = flair_slice(i:min(i+sZ-1, height), j:min(j+sZ-1, width));
            
            % Get bounds for that data
            img = flair_slice(i:min(i+sZ-1, height), j:min(j+sZ-1, width));
            
            % Check the patches are of correct dimensions, otherwise add 0s
            % to the bottom/right
            if size(img, 1) ~= sZ
                img = padarray(img, sZ-size(img,1), 0, "post"); % add padding to the right
            end
            if size(img, 2) ~= sZ
                img = padarray(img, [0 sZ-size(img,2)], 0, "post"); % add padding to the bottom
            end

            y = net.predict(img);
            y = double(y >= 0);
            
            % Get verification data
            if strcmp(transType, "BiasField")
                dataPath = "reach_monai_"+transType+"_"+string(sZ)+"_"...
                +sbName+"_"+string(c)+"_"+string(i)+"_"+string(j)+"_3_"...
                +epsilon+"_"+nPix+"_relax-star-range0.95.mat";
            else
            dataPath = "reach_monai_"+transType+"_"+string(sZ)+"_"...
                +sbName+"_"+string(c)+"_"+string(i)+"_"+string(j)+"_"...
                +epsilon+"_"+nPix+"_relax-star-range0.95.mat";
            end

            if any(contains(resFiles,dataPath))

                reachData = load("results/"+dataPath);
                lb = reachData.lb;
                ub = reachData.ub;
                verTime = verTime + reachData.rT;
        
                % 1) get correctly classified as 0 (background)
                ver_background = (ub <= 0);
                ver_background = ~ver_background;
                
                % 2) get correctly classified as 1 (lession)
                ver_lesion = (lb > 0);

                % 3) get verified output
                ver_img = 2*ones(size(img)); % pixel = 2 -> unknown
                
                background = find(ver_background == 0); % 0
                ver_img(background) = 0;
                
                lesion = find(ver_lesion == 1); % 1
                ver_img(lesion) = 1;

            else

                ver_img = y; % if slice contains no white matter

            end
            
            % Assign local data to main variable
            if (i+sZ-1) > height
                endH = height;
            else
                endH = i+sZ-1;
            end
            if (j+sZ-1) > width
                endW = width;
            else
                endW = j+sZ-1;
            end

            pred_c(i:endH,j:endW) = y(1:(1+endH-i),1:(1+endW-j));
            ver_c(i:endH,j:endW) = ver_img(1:(1+endH-i),1:(1+endW-j));
            
        end

    end

end