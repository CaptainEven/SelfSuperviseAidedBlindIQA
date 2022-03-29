%% setup
clear; clc;
addpath(genpath('code_imdistort'));

%% set paths
ref_dir = 'g:/ref_plates/'
csv_name = 'plates_ref_imgs.csv'
csv_path = [ref_dir csv_name]
dist_dir = 'g:/dist_plates'


%% read the info of pristine images
tb = readtable(csv_path);  %% 'kadis700k_ref_imgs.csv'
tb = table2cell(tb);


%% generate distorted images in dist_imgs folder
for i = 1:size(tb, 1)
    im_path = [ref_dir tb{i, 1}]
    if exist(im_path,'file')
        ref_im = imread(im_path);  %% 'ref_imgs/'
        disp([im_path ' read.'])
    else
        disp([im_path 'not eixt!']);
    end
    dist_type = tb{i, 2};
    
    for dist_level = 1:5
        [dist_im] = imdist_generator(ref_im, dist_type, dist_level);
        strs = split(tb{i, 1},'.');
        dist_im_path = [strs{1}  '_' num2str(tb{i,2}, '%02d')  '_' num2str(dist_level, '%02d') '.bmp'];
        disp(dist_im_path);
        imwrite(dist_im, [dist_dir dist_im_path]);  % 'dist_imgs/'
    end
    
    
end







