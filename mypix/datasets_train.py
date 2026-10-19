import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_img=None, transforms_label=None, mode='train'):
        # transforms.Compose()的作用是把transforms_img中所有的步骤整合到一起
        self.transform_A = transforms.Compose(transforms_img)
        self.transform_B = transforms.Compose(transforms_label)
        # glob.glob()返回所有匹配的文件路径列表
        self.files = sorted(glob.glob(os.path.join(root, mode) + '/*.*'))

    # 在用 for..in.. 迭代对象时,如果对象没有实现__iter__ __next__迭代器协议,Python的解释器就会去寻找__getitem__ 来迭代对象,
    #                                                  如果连__getitem__ 都没有定义，这解释器就会报对象不是迭代器的错误
    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w/2, h))
        img_B = img.crop((w/2, 0, w, h))
        img_B = img_B.convert('1')

        img_A = self.transform_A(img_A)
        img_B = self.transform_B(img_B)

        return {'A': img_A, 'B': img_B}

    def __len__(self):
        return len(self.files)
