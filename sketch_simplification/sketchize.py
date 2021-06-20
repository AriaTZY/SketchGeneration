# This is the pipeline test, you can fine tuning some parameters, and observe the output
# Then pick up a optimal parameter set you satisfied
import torch
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.serialization import load_lua

from PIL import Image
import argparse
import cv2 as cv
import matplotlib.pylab as plt
import os
import numpy as np
import sys

# use_cuda = torch.cuda.device_count() > 0


def load_model(model_path):
    # parse parameters
    parser = argparse.ArgumentParser(description='Sketch simplification demo.')
    # parser.add_argument('--model', type=str, default='model_gan.t7', help='Model to use.')
    parser.add_argument('--img',   type=str, default='data/pencil.png',     help='Input image file.')
    parser.add_argument('--out',   type=str, default='data/sketch.png',      help='File to output.')
    opt = parser.parse_args()

    # load model
    cache = load_lua(model_path, long_size=8)
    model = cache.model
    immean = cache.mean
    imstd = cache.std
    model.evaluate()

    return opt, model, immean, imstd


def model_simplify(opt, model, immean, imstd, use_cuda=False):

    data = Image.open(opt.img).convert('L')
    # data = Image.open('pencil.png').convert('L')
    w, h = data.size[0], data.size[1]
    pw = 8-(w % 8) if w % 8 != 0 else 0
    ph = 8-(h % 8) if h % 8 != 0 else 0
    data = ((transforms.ToTensor()(data)-immean)/imstd).unsqueeze(0)
    if pw != 0 or ph != 0:
       data = torch.nn.ReplicationPad2d((0, pw, 0, ph))(data).data
    if use_cuda:
       pred = model.cuda().forward(data.cuda()).float()
    else:
       pred = model.forward(data)
    save_image(pred[0], opt.out)


def pencil2sketch(img, model_path, use_cuda):
    # ===========================================
    # Sketch transformation
    # ===========================================
    args = load_model(model_path)
    cv.imwrite(args[0].img, img)
    model_simplify(*args, use_cuda)
    sketch = cv.imread(args[0].out, 1)
    return sketch

