import argparse
import os
import numpy as np
import time
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
from networks import define_G
from datasets_predict import ImageDataset
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default="Nuclei_Segmentation", help='name of the dataset')
parser.add_argument('--n_cpu', type=int, default=6, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=512, help='size of image height')
parser.add_argument('--img_width', type=int, default=512, help='size of image width')
parser.add_argument('--mod', type=int, default=3, help='generator saved index')
parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of residual blocks in generator')
parser.add_argument('--path', type=str, default='/media/sgu/软件/PythonProjects/gujiarui/', help='path to code and data')
opt = parser.parse_args()
print(opt)

def sample_images(imgs, model, path, idx=None,save=False):
    """Saves a generated sample from the test set"""
    fake_B = model(Variable(imgs['A'].type(Tensor)))
    if save:
        save_image(fake_B, os.path.join(path, 'predict_results/pred_map', str(idx)) +'.jpg', normalize=True)
    return fake_B

transforms_img = [transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.666, 0.465, 0.615), (0.179, 0.191, 0.154))]

predict_dataloader = DataLoader(ImageDataset(os.path.join(opt.path, "Data/%s" % opt.dataset_name), transforms_img=transforms_img, mode="predict"),
                          batch_size=1, shuffle=False, num_workers=1)

cuda = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

generator = define_G(input_nc=3, output_nc=1, ngf=64)

if cuda:
    generator = torch.nn.DataParallel(generator).cuda()

generator.load_state_dict(torch.load(os.path.join(opt.path, 'saved_model/Nuclei_Segmentation/generator_30.pth')))
generator.eval()

def main():
    elapse = []
    for i, image in enumerate(predict_dataloader):
        print('Predicting image ', i)
        start = time.process_time()
        sample_images(image, generator, opt.path, idx=i, save=True)
        elapsed = (time.process_time() - start)
        elapse.append(elapsed)
        print(elapsed)
    print('Average testing time is ', np.mean(elapse))

if __name__ == "__main__":
    main()
