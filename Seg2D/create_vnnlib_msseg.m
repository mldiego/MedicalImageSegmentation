function create_vnnlib_msseg(image, attack)
% Based on image and attack (bright, dark, linf), create the vnnlib
% specification: input (lb,ub), and output spec (halfspace)



end


function Hs = mask2spec(target)
% Convert mask target of MSSEG16 data to specification
% target = image mask (0, or 1) -> class target for each pixel
% Hs = halfspace that encodes the output property


end


function [lb,ub] = createInputBounds(image, attack)
% Create input bounds for the vnnlib (vector form)



end