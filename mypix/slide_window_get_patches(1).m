clear;
clc;

rgb_folder = '/media/sgu/软件/PythonProjects/gujiarui/dataset/RGBimages/';
rgb_names = dir([rgb_folder '*.jpg']);
train_folder = '/media/sgu/软件/PythonProjects/gujiarui/dataset/file/';
mask_folder = '/media/sgu/软件/PythonProjects/gujiarui/dataset/binaryMasks/';
mask_names = dir([mask_folder '*.png']);
label_folder = '/media/sgu/软件/PythonProjects/gujiarui/dataset/mask/';

index = 1;

for i = 1:length(rgb_names)
    disp(i);
    rgb_path = [rgb_folder rgb_names(i).name];
    rgb_data = imread(rgb_path);
    mask_path = [mask_folder mask_names(i).name];
    mask_data = imread(mask_path);
    for k = 1:122:489
        for kk = 1:122:489
            patch_rgb = rgb_data(k:k+511, kk:kk+511, :);
            patch_label = mask_data(k:k+511, kk:kk+511, :);
            imwrite(patch_rgb, [train_folder, num2str(index, '%05d'), '.png']);
            imwrite(patch_label, [label_folder, num2str(index, '%05d'), '.png']);
            index = index+1;
            if (k+511)>1000
                break;
            end
            if (kk+511)>1000
                break;
            end
        end
    end
end