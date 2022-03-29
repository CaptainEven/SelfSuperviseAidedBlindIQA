%% setup
clear; clc;
addpath(genpath('code_imdistort'));


%% read the info of pristine images
tb = readtable('kadis700k_ref_imgs.csv');
tb = table2cell(tb);


%% generate distorted images in dist_imgs folder
for i = 1:size(tb,1)
    ref_im = imread(['ref_imgs/' tb{i,1}]);
    dist_type = tb{i,2};
    
    for dist_level = 1:5
        [dist_im] = imdist_generator(ref_im, dist_type, dist_level);
        strs = split(tb{i,1},'.');
        dist_im_name = [strs{1}  '_' num2str(tb{i,2},'%02d')  '_' num2str(dist_level,'%02d') '.bmp'];
        disp(dist_im_name);
        imwrite(dist_im, ['dist_imgs/' dist_im_name]);
    end
    
    
end







