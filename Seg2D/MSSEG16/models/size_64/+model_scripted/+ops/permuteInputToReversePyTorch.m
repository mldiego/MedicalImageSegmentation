function [dlXPerm, MATLABFormat] = permuteInputToReversePyTorch(dlX, rank)
% Function to determine the permutation vectors from DLT format to
% reverse-Python, for a given labeled dlarray dlX. Permutes 
% dlX into reverse-Python dimension ordering, returning the result as dlXPerm 
% and the original MATLAB Format. The result will be unlabeled. U-labelled
% inputs are considered to in forward PyTorch Ordering.

%   Copyright 2023 The MathWorks, Inc.


MATLABFormat = dims(dlX); 
if isdlarray(dlX) && ~isempty(dims(dlX))
    if all(dims(dlX) == 'U')
        dlXPerm = permute(stripdims(dlX), fliplr(1:max(2,rank)));
        dlXPerm = dlarray(dlXPerm, repmat('U',1,max(2,rank)));
    else   
        switch MATLABFormat
            case 'SSCB'
                permDLTToReversePython = [2 1 3 4]; % HWCN -> WHCN
            case 'SSC'
                permDLTToReversePython = [2 1 3]; % HWC -> WHC     
            case 'SSSCB'
                % NOTE: Although fwd-PyTorch canonically uses the format "NCDHW",
                % we choose to preserve the order of the spatial dimensions,
                % treating them as HWD rather than DHW.
                permDLTToReversePython = [3 2 1 4 5]; % HWDCN -> DWHCN     
            case 'SSSC'
                permDLTToReversePython = [3 2 1 4]; % HWDC -> DWHC         
            case 'CB'
                permDLTToReversePython = [1 2]; % CN -> CN
            case 'SCBT'
                permDLTToReversePython = [1 4 2 3]; %SCBT -> STCB
            otherwise
                throwAsCaller(MException(message("nnet_cnn_pytorchconverter:pytorchconverter:DlarrayFormatNotRecognized"...
            ,MATLABFormat)));
        end
        dlXPerm = permute(stripdims(dlX), permDLTToReversePython);
        dlXPerm = dlarray(dlXPerm, repmat('U',1,max(2,ndims(dlX))));
    end
    
else
    dlXPerm = dlX;
    MATLABFormat = [];
end

end