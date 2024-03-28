function validateInput(dlX,rank)
%Function that validates the input dlarray is in one of the accepted
%dlarray formats. It also makes sure there is a valid rank input if
%U-labelled dlarray is passed.

%   Copyright 2023 The MathWorks, Inc.

if isdlarray(dlX)
    MATLABFormat = dims(dlX);
    if ~any(dims(dlX) == 'U')
        switch MATLABFormat
            case {'SSCB','SSC','SSSCB','SSSC','CB','SCBT'}
                %Valid input Labels
            case {'CBT'}
                %CBT format inputs are not accepted when formats are not propagated
                %to the first layer of the network
                throwAsCaller(MException(message("nnet_cnn_pytorchconverter:pytorchconverter:DlarrayFormatNotRecognized",MATLABFormat)));
            otherwise
                throwAsCaller(MException(message("nnet_cnn_pytorchconverter:pytorchconverter:DlarrayFormatNotRecognized",MATLABFormat)));
        end
    else
        % dlX is in forward-PyTorch dimension order.
        %Rank is required to prevent dropping of trailing singleton
        %dimensions
        if isempty(rank)
            throwAsCaller(MException(message('nnet_cnn_pytorchconverter:pytorchconverter:LossOfRankError')));
        end
    end
end

