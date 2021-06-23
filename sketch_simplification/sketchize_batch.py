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

""" python sketchize_batch.py --cuda True --model ../../sketch_simplification-master/model_gan.t7 --input ../../Dataset/Test/pencil/ --out ../../Dataset/Test/sketch/ """
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sketch simplification demo.')
    parser.add_argument('--model', type=str, default='model_gan.t7', help='Model to use.')
    parser.add_argument('--img_sz', type=int, default=600, help='The size you want to resize the input image')
    parser.add_argument('--input',   type=str, default='../data/dataset/pencil/',     help='Input image file.')
    parser.add_argument('--out',   type=str, default='../data/dataset/sketch/',      help='File to output.')
    parser.add_argument('--cuda',   type=bool, default=False, help='whether to use cuda')
    opt = parser.parse_args()

    # validation check
    assert os.path.exists(opt.input), 'ERROR: Make sure you already create pencil file folder'
    if not os.path.exists(opt.out): os.makedirs(opt.out)

    # opt.model = '../../sketch_simplification-master/model_gan.t7'

    print('USE CUDA?', opt.cuda)

    width = opt.img_sz
    max_batch_num = 2
    start_index = 0  # this is set for in case you are resuming to generate sketches

    cache  = load_lua( opt.model , long_size=8)
    model  = cache.model
    immean = cache.mean
    imstd  = cache.std
    model.evaluate()

    img_path = opt.input
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

        if opt.cuda:
           pred = model.cuda().forward(data.cuda()).float()
        else:
           pred = model.forward(data)

        # save and post process
        out_path = opt.out
        for i in range(batch_size):
            img_name = out_path + img_list[batch_idx*max_batch_num + i + start_index]
            save_image(pred[i], img_name)


