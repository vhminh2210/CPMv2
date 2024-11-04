import os
import copy
from parser import get_args
from tqdm import tqdm

import matplotlib.pyplot as plt

import cv2
import numpy as np
import torch.optim
from makeup import Makeup
from PIL import Image

import torch
import random
import functools
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models

from SemCorr import *

args = get_args()

cpmodel = CPMPlus(args)

src_list = os.listdir(args.src_dir)
ref_list = os.listdir(args.ref_dir)

txt_dir = '/content/gdrive/MyDrive/AIEngineer/CPM_v2/makeup_dataset/uv/txt'
pos_dir = '/content/gdrive/MyDrive/AIEngineer/CPM_v2/makeup_dataset/uv/pos'

optimizer = torch.optim.Adam(cpmodel.scm.parameters(), lr= args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= args.max_iter)

training_loss = []
device = torch.device('cpu')
if args.device == 'cuda':
    device = torch.device('cuda:0')

cpmodel = cpmodel.to(device)

print('Training started!')

for t in range(args.max_iter):
    idxA = random.randint(0, len(src_list) - 1)
    idxB = random.randint(0, len(ref_list) - 1)

    txtA = np.array(Image.open(os.path.join(txt_dir, 'non-makeup', src_list[idxA])))
    posA = np.array(Image.open(os.path.join(pos_dir, 'non-makeup', src_list[idxA])))

    txtB = np.array(Image.open(os.path.join(txt_dir, 'makeup', ref_list[idxB])))
    posB = np.array(Image.open(os.path.join(pos_dir, 'makeup', ref_list[idxB])))

    txtA, _ , txtB, _ = cpmodel.uvTransform(txtA, posA, txtB, posB)
    txtA, txtB = txtA.to(device), txtB.to(device)

    optimizer.zero_grad()

    # print(txtA, txtB)

    loss = cpmodel.calLoss(txtA, txtB)
    loss.backward()
    training_loss.append(loss.item())

    optimizer.step()
    scheduler.step()

    print(loss.item())

    if t % 10 == 0:
      print(f'[Iter {t} / {args.max_iter}]: Loss = {np.round(loss.item(), 4)}')
    