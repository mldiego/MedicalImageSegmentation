
% Get model 1
ptmodel = importNetworkFromPyTorch('model_scripted.pt');
tanhL = tanhLayer;
ptmodel = replaceLayer(ptmodel, 'aten__tanh4', tanhL);
inputLayer = imageInputLayer([64 64 1]); 
ptmodel = addInputLayer(ptmodel, inputLayer, Initialize=1);

% Model 2
onnxmodel = importONNXNetwork("best_model.onnx");
inputSize = onnxmodel.Layers(1).InputSize;

% Counter example file
% cename = "img_297_sliceSize_64_linf_pixels_10_eps_0.0001_region.txt";
% [x,y] = getCounterExample(cename);


%% 
x = reshape(x, inputSize);
% x1 = permute(x, [2 1]);

% Compute output predictions
y1 = predict(ptmodel, x);
% y1 = permute(y1, [2 1])';
y2 = predict(onnxmodel, x);

all(y1 == y2, 'all')

disp([y(55) y1(55), y2(55)])

%% Helper functions
function [x,y] = getCounterExample(cename)
    fid = fopen(cename, "r");
    fgetl(fid); 
    fgetl(fid);
    fgetl(fid);
    % First three lines do nothing
    x = [];
    y = [];
    tline = fgetl(fid);
    while tline ~= ")" % end of file
        a = split(tline, " ");
        a = a{end}; 
        a = split(a, ")");
        a = a{1};
        a = str2double(a);
        if contains(tline, "X")
            x = [x ; a];
        else
            y = [y ; a];
        end
        tline = fgetl(fid);
    end

end