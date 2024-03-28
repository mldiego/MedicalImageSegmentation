function [dlXPerm, MATLABFormat,permReversePythonToDLT] = permuteToReversePyTorch(dlX, forwardPyTorchFormat, rank)
% Function to determine the permutation vectors from DLT format to
% reverse-Python and vice-versa, for a given labeled dlarray dlX. Permutes 
% dlX into reverse-Python dimension ordering, returning the result as dlXPerm.
% The result will be unlabeled.

%   Copyright 2022-2023 The MathWorks, Inc.

MATLABFormat = dims(dlX);

if isdlarray(dlX) && ~isempty(dims(dlX)) && ~any(dims(dlX) == 'U') && forwardPyTorchFormat ~= ""
    dlXRank = ndims(dlX);
    if contains(forwardPyTorchFormat, "*") && dlXRank == strlength(string(forwardPyTorchFormat))
       forwardPyTorchFormat = replace(forwardPyTorchFormat, "*", "B");
    else
       forwardPyTorchFormat = erase(forwardPyTorchFormat, "*");
    end
    switch forwardPyTorchFormat
        case {'BCSS'} %from SSCB
            iValidatePropagatedLabelAsExpected(MATLABFormat, "SSCB");
            permDLTToReversePython = [2 1 3 4]; % HWCN -> WHCN
            permReversePythonToDLT = [2 1 3 4]; % WHCN -> HWCN
        case {'CSS'} %from SSC
            iValidatePropagatedLabelAsExpected(MATLABFormat, "SSC");
            permDLTToReversePython = [2 1 3]; % HWC -> WHC
            permReversePythonToDLT = [2 1 3]; % WHC -> HWC
        case {'BCSSS'} %from SSSCB
            % NOTE: Although fwd-PyTorch canonically uses the format "NCDHW",
            % we choose to preserve the order of the spatial dimensions,
            % treating them as HWD rather than DHW.
            iValidatePropagatedLabelAsExpected(MATLABFormat, "SSSCB");
            permDLTToReversePython = [3 2 1 4 5]; % HWDCN -> DWHCN
            permReversePythonToDLT = [3 2 1 4 5]; % DWHCN -> HWDCN 
        case {'CSSS'} %from SSSC
            iValidatePropagatedLabelAsExpected(MATLABFormat, "SSSC");
            permDLTToReversePython = [3 2 1 4]; % HWDC -> DWHC
            permReversePythonToDLT = [3 2 1 4]; % DWHC -> HWDC 
        case {'BC'} %from CB
            iValidatePropagatedLabelAsExpected(MATLABFormat, "CB");
            permDLTToReversePython = [1 2]; % CN -> CN
            permReversePythonToDLT = [1 2]; % CN -> CN
        case {'BCT'} % from CBT
            iValidatePropagatedLabelAsExpected(MATLABFormat, "CBT");
            permDLTToReversePython = [3 1 2]; %CBT -> TCB
            permReversePythonToDLT = [2 3 1]; %TCB -> CBT
        case {'CT'} % from CT
            iValidatePropagatedLabelAsExpected(MATLABFormat, "CT");
            permDLTToReversePython = [2 1]; %CT -> TC
            permReversePythonToDLT = [2 1]; %TC -> CT
        case {'TBC'} %from CBT
            iValidatePropagatedLabelAsExpected(MATLABFormat, "CBT");
            permDLTToReversePython = [1 2 3]; %CBT -> CBT
            permReversePythonToDLT = [1 2 3]; %CBT -> CBT
        case {'TC'} %from CT
            iValidatePropagatedLabelAsExpected(MATLABFormat, "CT");
            permDLTToReversePython = [1 2]; %CT -> CT
            permReversePythonToDLT = [1 2]; %CT -> CT
        case {'BCTS'} %SCBT
            iValidatePropagatedLabelAsExpected(MATLABFormat, "SCBT");
            permDLTToReversePython = [1 4 2 3]; %SCBT -> STCB
            permReversePythonToDLT = [1 3 4 2]; %STCB -> SCBT
        case {'BCS'} %SCB
            iValidatePropagatedLabelAsExpected(MATLABFormat, "SCB");
            permDLTToReversePython = [1 2 3]; %SCB -> SCB
            permReversePythonToDLT = [1 2 3]; %SCB -> SCB
        case {'CTS'} %SCT
            iValidatePropagatedLabelAsExpected(MATLABFormat, "SCT");
            permDLTToReversePython = [1 3 2]; %SCT -> STC
            permReversePythonToDLT = [1 3 2]; %STC -> SCT
        otherwise 
            error(message('nnet_cnn_pytorchconverter:pytorchconverter:DlarrayFormatNotRecognized', forwardPyTorchFormat));
    end
    dlXPerm = permute(stripdims(dlX), permDLTToReversePython);
    dlXPerm = dlarray(dlXPerm, repmat('U',1,max(2,ndims(dlX))));
elseif any(dims(dlX) == 'U')
    dlXPerm = permute(stripdims(dlX), fliplr(1:max(2,rank)));
    dlXPerm = dlarray(dlXPerm, repmat('U',1,max(2,rank)));
    permReversePythonToDLT = []; %Not relevant since permutations happend from rev-pytorch to forward pytorch
else
    error(message('nnet_cnn_pytorchconverter:pytorchconverter:DlarrayFormatNotRecognized', "Unlabelled"));
end

    function iValidatePropagatedLabelAsExpected(runtimeLabel, expectedLabel)
        if(runtimeLabel ~= expectedLabel)
            error(message('nnet_cnn_pytorchconverter:pytorchconverter:DlArrayFormatMismatch', expectedLabel, runtimeLabel));
        end
    end

end