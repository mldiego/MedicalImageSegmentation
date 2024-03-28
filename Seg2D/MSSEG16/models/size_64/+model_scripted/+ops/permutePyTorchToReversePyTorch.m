function varargout = permutePyTorchToReversePyTorch(varargin)
% Function to permute data from PyTorch dimension ordering into the 
% reverse PyTorch dimension ordering. The inputs must be structs containing
% a 'value' and a 'rank' field. The input values must be either unlabelled,
% or labelled with only U's. The outputs will be structs containing
% U-labelled values. 

%   Copyright 2023 The MathWorks, Inc.

varargout = cell(1, nargin);
for i=1:nargin
    X = varargin{i};
    iVerifyInputStruct(X);   
    if X.rank >= 2
        X.value = permute(X.value, fliplr(1:X.rank));
    else
        % Always return a column vector for 1D inputs
        X.value = X.value(:);
    end
    X.value = dlarray(X.value, repmat('U', 1, max(2,X.rank)));
    varargout{i} = X;
end
end

function iVerifyInputStruct(X)
% Verifies that the input is in the expected format: a struct containing a 
% 'value' and 'rank' field. The value should be an unlabelled or U-labelled
% dlarray.
if ~(isstruct(X) &&...
        isfield(X, 'value') &&...
        isfield(X, 'rank') &&...
        ~isempty(X.rank) &&...
        isdlarray(X.value) &&...
        (dims(X.value)=="" || unique(dims(X.value))=="U"))
    error(message('nnet_cnn_pytorchconverter:pytorchconverter:PlaceholderPermuteFcnInputs'));
end
end