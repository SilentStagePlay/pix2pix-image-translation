import os
import random
import shutil

fileDir = '/media/sgu/软件/PythonProjects/gujiarui/Data/Nuclei_Segmentation/train/'
targDir = '/media/sgu/软件/PythonProjects/gujiarui/Data/Nuclei_Segmentation/val/'

pathDir = os.listdir(fileDir)
filenumber = len(pathDir)
rate = 0.3
picknumber = int(filenumber * rate)
sample = random.sample(pathDir, picknumber)
for name in sample:
    shutil.move(fileDir+name, targDir+name)