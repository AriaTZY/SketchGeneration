import torch
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.serialization import load_lua

from PIL import Image
import argparse
import os
import cv2 as cv
import numpy as np
import time

parser = argparse.ArgumentParser(description='Sketch simplification demo.')
parser.add_argument('--model', type=str, default='model_gan.t7', help='Model to use.')
parser.add_argument('--img',   type=str, default='data/input/',     help='Input image file.')
parser.add_argument('--out',   type=str, default='data/output/',      help='File to output.')
opt = parser.parse_args()

use_cuda = torch.cuda.device_count() > 0
# use_cuda = False
print('USE CUDA?', use_cuda)

width = 600
max_batch_num = 2
start_index = 0  # this is set for in case you are resuming to generate sketches

cache  = load_lua( opt.model , long_size=8)
model  = cache.model
immean = cache.mean
imstd  = cache.std
model.evaluate()

img_path = opt.img
img_list = os.listdir(img_path)
num = len(img_list)
num = num - start_index
batch_num = int(np.floor(num/max_batch_num))

start_time = time.time()
# max batch size is 10, just ensure the memory won't be filled out
for batch_idx in range(batch_num + 1):
    print('process batch {}/{}, total time cost: {:.3f} s'.format(batch_idx, batch_num+1, time.time()-start_time))
    tensor_img_ls = []
    batch_size = max_batch_num

    # the last bach normally is not equal to 10
    if batch_idx == batch_num:
        batch_size = num - batch_num * max_batch_num

    for i in range(batch_size):
        img_name = img_path + img_list[batch_idx*max_batch_num + i + start_index]
        img = Image.open(img_name).convert('L')
        img = img.resize((width, width))
        tensor_img = ((transforms.ToTensor()(img) - immean) / imstd).unsqueeze(0)
        tensor_img_ls.append(tensor_img)

    data = torch.cat(tensor_img_ls, 0)

    if use_cuda:
       pred = model.cuda().forward(data.cuda()).float()
    else:
       pred = model.forward(data)

    # save and post process
    out_path = opt.out
    for i in range(batch_size):
        img_name = out_path + img_list[batch_idx*max_batch_num + i + start_index]
        save_image(pred[i], img_name)


