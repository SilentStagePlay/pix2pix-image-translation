clear;
clc;
mask_fext = '*.png';
pred_fext = '*.png';
maskpath = '/media/sgu/软件/PythonProjects/gujiarui/dataset/testmask/';
predpath = '/media/sgu/软件/PythonProjects/gujiarui/dataset/test_predict/';
mask_info = dir([maskpath mask_fext]);
pred_info = dir([predpath pred_fext]);
mask_namelist = {mask_info.name};
pred_namelist = {pred_info.name};
for idx = 1:length(mask_namelist)
    disp(idx);
    mask = imread([maskpath, mask_namelist{idx}]);
    mask = imbinarize(mask);
    pred = imread([predpath, pred_namelist{idx}]);
    pred = rgb2gray(pred);
    pred = imbinarize(pred);
    jiao = mask & pred;
    bing = mask | pred;
    acc(idx) = sum(sum(mask == pred)) / (512*512);
    dice(idx) = sum(sum(jiao)) / sum(sum(bing));
end
    