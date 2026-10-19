clear;
clc;

RGB_fext = '*.png';
MASK_fext = '*.png';

RGB_PATH = '/media/sgu/软件/PythonProjects/gujiarui/dataset/file_aug/';
MASK_PATH = '/media/sgu/软件/PythonProjects/gujiarui/dataset/mask_aug/';
SAVE_path = '/media/sgu/软件/PythonProjects/gujiarui/dataset/final_aug/';

RGB_info = dir([RGB_PATH RGB_fext]);
MASK_info = dir([MASK_PATH MASK_fext]);

RGB_namelist = {RGB_info.name};
MASK_namelist = {MASK_info.name};

file_num = size(RGB_namelist,2);

for idx = 1:file_num
    disp(idx);
    RGB_image = imread([RGB_PATH,RGB_namelist{idx}]);
    MASK_image(:,:,1) = imread([MASK_PATH,MASK_namelist{idx}]);
    MASK_image(:,:,2) = imread([MASK_PATH,MASK_namelist{idx}]);
    MASK_image(:,:,3) = imread([MASK_PATH,MASK_namelist{idx}]);
    JOINT = [RGB_image MASK_image];
    imwrite(JOINT, [SAVE_path, num2str(idx, '%05d'), '.png']);
end