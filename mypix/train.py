import argparse
import os
import numpy as np
import time
import datetime
import sys
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from networks import define_G, PixelDiscriminator, weights_init_normal
from PIL import Image
from datasets_train import ImageDataset
import torch.nn as nn
import torch
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="Nuclei_Segmentation", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=15, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=6, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=512, help='size of image height')
parser.add_argument('--img_width', type=int, default=512, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=50, help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=5, help='interval between model checkpoints')
parser.add_argument('--path', type=str, default='/media/sgu/软件/PythonProjects/gujiarui/', help='path to code and data')

opt = parser.parse_args()
print(opt)


cuda = True if torch.cuda.is_available() else False

# 均方损失函数 loss(xi,yi)=(xi−yi)^2
# torch.nn.MSELoss(reduce=True,size_average=True)
# 如果size_average=True,返回loss.mean()
# 如果size_average=False,返回loss.sum()
# 二者默认为True
criterion_GAN = torch.nn.MSELoss()

# loss(xi,yi)=(xi−yi)^2
# torch.nn.L1Loss(reduce=False, size_average=False)
# 如果size_average=True,返回loss.mean()
# 如果size_average=False,返回loss.sum()
# 二者默认为True
criterion_pixelwise = torch.nn.L1Loss()

# Calculate output of image discriminator (PatchGAN)
# //是求整商
# **是幂运算
patch = (1, 32, 32)
# 初始化生成器和判别器
generator = define_G(input_nc=3, output_nc=1, ngf=64)
discriminator = PixelDiscriminator(input_nc=4)

if cuda:
    # torch.nn.DataParallel(model)默认所有存在的显卡都会被使用(多用于多GPU的情况)
    generator = torch.nn.DataParallel(generator).cuda()
    discriminator = torch.nn.DataParallel(discriminator).cuda()

    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

if opt.epoch != 0:
    # 加载预训练模型
    generator.load_state_dict(torch.load(os.path.join(opt.path,'/saved_model/%s/generator_%d.pth' % (opt.dataset_name, opt.epoch))))
    discriminator.load_state_dict(torch.load(os.path.join(opt.path,'/saved_model/%s/discriminator_%d.pth' % (opt.dataset_name, opt.epoch))))
else:
    # 初始化参数
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Configure dataloaders and Data Augmentation
transforms_img = [transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.666, 0.465, 0.615), (0.179, 0.191, 0.154))]

transforms_label = [transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                transforms.ToTensor()]

transforms_val_img = [transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.666, 0.465, 0.615), (0.179, 0.191, 0.154))]

transforms_val_label = [transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                transforms.ToTensor()]

dataloader = DataLoader(ImageDataset(os.path.join(opt.path, "Data/%s" % opt.dataset_name),
                                     transforms_img=transforms_img,
                                     transforms_label=transforms_label,
                                     mode='train'),
                        batch_size=opt.batch_size,
                        shuffle=True,
                        num_workers=opt.n_cpu)
print('len of train batch is: ', len(dataloader))

val_dataloader = DataLoader(ImageDataset(os.path.join(opt.path, "Data/%s" % opt.dataset_name),
                                         transforms_img=transforms_val_img,
                                         transforms_label=transforms_val_label,
                                         mode='val'),
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=1)
print('len of val batch is: ', len(val_dataloader))

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def sample_images(batches_done, path):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs['A'].type(Tensor))
    real_B = Variable(imgs['B'].type(Tensor))
    fake_B = generator(real_A)
    os.makedirs(os.path.join(path,'images/%s/img/' % opt.dataset_name), exist_ok=True)
    os.makedirs(os.path.join(path,'images/%s/gt/' % opt.dataset_name), exist_ok=True)
    os.makedirs(os.path.join(path,'images/%s/pred/' % opt.dataset_name), exist_ok=True)
    save_image(real_A, os.path.join(path,'images/%s/img/%s_image.png' % (opt.dataset_name, batches_done)), nrow=3, normalize=True, scale_each=True)
    save_image(real_B, os.path.join(path,'images/%s/gt/%s_image.png' % (opt.dataset_name, batches_done)), nrow=3, normalize=True, scale_each=True)
    save_image(fake_B, os.path.join(path,'images/%s/pred/%s_image.png' % (opt.dataset_name, batches_done)), nrow=3, normalize=True, scale_each=True)

prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    # enumerate()用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在for循环当中
    for i, batch in enumerate(dataloader):

        # 网络输入
        # torch.autograd.Variable是Autograd的核心类,它封装了Tensor,并整合了反向传播的相关实现
        # torch.autograd.Variable(tensor, requires_grad = True)用来定义Variable,requires_grad用来指定是否需要计算梯度,默认为False
        # Varibale包含三个属性： data：存储了Tensor,是本体的数据
        #                     grad：保存了data的梯度,本体是个Variable而非Tensor,与data形状一致
        #                     grad_fn：指向Function对象,用于反向传播的梯度计算之用

        real_A = Variable(batch['A'].type(Tensor))
        real_B = Variable(batch['B'].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        # 将梯度置0,也就是把loss关于weight的导数变成0
        optimizer_G.zero_grad()

        # GAN loss
        fake_B = generator(real_A)
        pred_fake = discriminator(real_A, fake_B)
        loss_GAN = criterion_GAN(pred_fake, valid)
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_B, real_B)

        # Loss weight of L1 pixel-wise loss between translated image and real image
        lambda_pixel = 0.99

        # Total loss
        loss_G = (1-lambda_pixel)*loss_GAN + lambda_pixel * loss_pixel

        loss_G.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(real_A,real_B)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(real_A, fake_B.detach())
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        # loss.backward()反向传播loss
        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s" %
                                                        (epoch, opt.n_epochs,
                                                        i, len(dataloader),
                                                        loss_D.item(), loss_G.item(),
                                                        loss_pixel.item(), loss_GAN.item(),
                                                        time_left))

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done, opt.path)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), os.path.join(opt.path, 'saved_model/%s/generator_%d.pth' % (opt.dataset_name, epoch)))
        torch.save(discriminator.state_dict(), os.path.join(opt.path, 'saved_model/%s/discriminator_%d.pth' % (opt.dataset_name, epoch)))