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

import matplotlib.pyplot as plt
from torchvision.utils import save_image
from PIL import Image

import seaborn as sns

path = '/chenxue/experiment_data/anomaly_diffusion_all_train_brats2021_64x64/counterfactual_sampling_gt/samples.npz'
file_data = np.load(path)
for key, data in file_data.item():
	print(key, ":", type(data))

