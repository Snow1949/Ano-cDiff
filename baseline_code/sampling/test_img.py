"""
Like score_sampling.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""
import os
import numpy as np
import torch as th
import torch.distributed as dist
from pathlib import Path
import sys
sys.path.append(str(Path.cwd()))
import argparse
import random
import cv2

import matplotlib.pyplot as plt
from torchvision.utils import save_image
from PIL import Image

import seaborn as sns

img1 = plt.imread("/chenxue/Diff-SCM-main/exogenous_noise/latent_1.png")
img2 = plt.imread("/chenxue/Diff-SCM-main/exogenous_noise/latent_2.png")

err = cv2.absdiff(img1, img2)

# hot = sns.heatmap(img2, xticklabels=False,yticklabels=False,cmap="coolwarm",cbar=False)

fig = plt.figure('result')
plt.axis('off')  #关闭坐标轴

plt.subplot(2,2,1)     #将窗口分为两行两列四个子图，则可显示四幅图片
plt.title('img1')   #第一幅图片标题
plt.imshow(img1)

plt.subplot(2,2,2)
plt.title('img2')
plt.imshow(img2)

plt.subplot(2,2,3)
plt.title('imgerr')
plt.imshow(err)

plt.subplot(2,2,4)
plt.title('imghot')
plt.imshow(err)

fig.tight_layout()#调整整体空白
plt.subplots_adjust(wspace =0)#调整子图间距
plt.show()   #显示

plt.savefig('/chenxue/Diff-SCM-main/exogenous_noise/0.png')
