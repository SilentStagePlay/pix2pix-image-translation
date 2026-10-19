clear;
clc;

rgb_folder = './RGBimages/';
rgb_names = dir([rgb_folder '*.jpg']);
train_folder = './file/';
mask_folder = './binaryMasks/';
mask_names = dir([mask_folder '*.png']);
label_folder = './mask/';

index = 1;

for i = 1:length(rgb_names)
    disp(i);
    rgb_path = [rgb_folder rgb_names(i).name];
    rgb_data = imread(rgb_path);
    mask_path = [mask_folder mask_names(i).name];
    mask_data = imread(mask_path);
    image1 = rgb_data(1:512, 1:512, :);
    mask1 = mask_data(1:512, 1:512);
    image2 = rgb_data(1:512, 244:755, :);
    mask2 = mask_data(1:512, 244:755);
    image3 = rgb_data(1:512, 489:1000, :);
    mask3 = mask_data(1:512, 489:1000);
    image4 = rgb_data(244:755, 1:512, :);
    mask4 = mask_data(244:755, 1:512);
    image5 = rgb_data(244:755, 244:755, :);
    mask5 = mask_data(244:755, 244:755);
    image6 = rgb_data(244:755, 489:1000, :);
    mask6 = mask_data(244:755, 489:1000);
    image7 = rgb_data(489:1000, 1:512, :);
    mask7 = mask_data(489:1000, 1:512);
    image8 = rgb_data(489:1000, 244:755, :);
    mask8 = mask_data(489:1000, 244:755);
    image9 = rgb_data(489:1000, 489:1000, :);
    mask9 = mask_data(489:1000, 489:1000);
    imwrite(image1,[train_folder, num2str(9*i-8, '%05d'), '.png']);
    imwrite(image2,[train_folder, num2str(9*i-7, '%05d'), '.png']);
    imwrite(image3,[train_folder, num2str(9*i-6, '%05d'), '.png']);
    imwrite(image4,[train_folder, num2str(9*i-5, '%05d'), '.png']);
    imwrite(image5,[train_folder, num2str(9*i-4, '%05d'), '.png']);
    imwrite(image6,[train_folder, num2str(9*i-3, '%05d'), '.png']);
    imwrite(image7,[train_folder, num2str(9*i-2, '%05d'), '.png']);
    imwrite(image8,[train_folder, num2str(9*i-1, '%05d'), '.png']);
    imwrite(image9,[train_folder, num2str(9*i, '%05d'), '.png']);
    imwrite(mask1,[label_folder, num2str(9*i-8, '%05d'), '.png']);
    imwrite(mask2,[label_folder, num2str(9*i-7, '%05d'), '.png']);
    imwrite(mask3,[label_folder, num2str(9*i-6, '%05d'), '.png']);
    imwrite(mask4,[label_folder, num2str(9*i-5, '%05d'), '.png']);
    imwrite(mask5,[label_folder, num2str(9*i-4, '%05d'), '.png']);
    imwrite(mask6,[label_folder, num2str(9*i-3, '%05d'), '.png']);
    imwrite(mask7,[label_folder, num2str(9*i-2, '%05d'), '.png']);
    imwrite(mask8,[label_folder, num2str(9*i-1, '%05d'), '.png']);
    imwrite(mask9,[label_folder, num2str(9*i, '%05d'), '.png']);
end