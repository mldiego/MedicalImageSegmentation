function [dlXOut] = labelWithPropagatedFormats(dlX,formats)
% Function to determine the label a dlarray with the propagated format.
% The U-labelled dlarray is assumed to be in reverse python format. Permute
% to DLT format and label. 
% Since batch is optional in PyTorch, we use rank with propagated label to 
% determine the format and permutations 

%   Copyright 2022-2023 The MathWorks, Inc.

dlXRank = dlX.rank;
if all(dims(dlX.value) =='U')
     if contains(formats, "*") && dlXRank == strlength(string(formats))
       formats = replace(formats, "*", "B");
    else
       formats = erase(formats, "*");
     end

    switch formats
        case {'CSS'}
            permReversePythonToDLT       = [2 1 3];
            outputFormat                 = 'SSC';
        case {'BCSS'}
            permReversePythonToDLT       = [2 1 3 4];
            outputFormat                 = 'SSCB';
        case {'CSSS'}
            permReversePythonToDLT       = [3 2 1 4];
            outputFormat                 = 'SSSC';
        case {'BCSSS'}
            permReversePythonToDLT       = [3 2 1 4 5];
            outputFormat                 = 'SSSCB';
        case {'BC'}
            permReversePythonToDLT       = [1 2];
            outputFormat                 = 'CB';
        case {'BCT'}
            permReversePythonToDLT       = [2 3 1]; %Pytorch: BCT || Rev-Pytorch: TCB
            outputFormat                 = 'CBT';
        case {'CT'}
            permReversePythonToDLT       = [1 2];
            outputFormat                 = 'CT';
        otherwise
            error('Unknown format and rank %s', formats);
    end
    
    dlXValue    = dlX.value;
    dlXPermuted = permute(stripdims(dlXValue),permReversePythonToDLT);
    dlXOut.value = dlarray(dlXPermuted,outputFormat);
    dlXOut.rank = dlXRank;
else
    dlXOut = dlX;
end

end
