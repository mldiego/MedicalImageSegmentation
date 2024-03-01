%% Example of how to load and process data from nifti files

pth2data = "../data/UMCL/subjects/patient01/1/";

flair = niftiread(pth2data+"flair.nii");
mask = niftiread(pth2data+"mask.nii");
% wm_mask = niftiread(pth2data+"wm_mask.nii"); % not there yet for umcl

% get max and min values for visualizing
iMax = max(flair, [], 'all');
iMin = min(flair, [], 'all');

% Visualize images
% bb = volshow(flair);
% dd = volshow(mask);

% Then for each of this we can take some slices of slices
% The 2D slices are 512x512...
% How do we choose which slices to analyze from each subject?
% How many slices per subject is good enough?


% Visualize a few examples

% % Around the start
% f3 = flair(3, :, :);
% m3 = mask(3, :, :);
% 
% figure;
% imshow(squeeze(f3), [min(f3, [], 'all'), max(f3, [], 'all')]);
% 
% figure;
% imshow(squeeze(m3));

% Around the middle
f100 = squeeze(flair(100, :, :));
m100 = squeeze(mask(100, :, :));

figure;
imshow(squeeze(f100), [min(f100, [], 'all'), max(f100, [], 'all')]);

figure;
imshow(squeeze(m100));

% % Around the end
% f190 = flair(190, :, :);
% m190 = mask(190, :, :);
% 
% figure;
% imshow(squeeze(f190), [min(f190, [], 'all'), max(f190, [], 'all')]);
% 
% figure;
% imshow(squeeze(m190));


% Clearly, we need to get slices around the middle...
win = centerCropWindow2d(size(f100), [64 64]);

img_slice = imcrop(f100, win);
mask_slice = imcrop(m100, win);

figure;
imshow(img_slice, [iMin, iMax]);

figure;
imshow(mask_slice);


% This one is okay, but we could use the white matter mask to only use
% slices that contain interesting bits rather than using some images that
% are all background


