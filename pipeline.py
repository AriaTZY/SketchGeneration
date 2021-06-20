# Pipeline process, from a world image to a sketch in vector form
from image2pencil.pencilize import image2pencil
from sketch_simplification.sketchize import pencil2sketch
from virtual_sketching.test_vectorization import sketch2vector
from virtual_sketching.my_vectorization import canvas_draw
import argparse
import os
import cv2 as cv
import matplotlib.pylab as plt


def display():
    plt.figure(figsize=(14, 5))
    plt.subplot(131)
    plt.imshow(img)
    plt.title('world image')
    plt.axis('off')

    # pencil = cv.Canny(pencil, 20, 200)
    plt.subplot(132)
    plt.imshow(pencil, cmap='gray')
    plt.title('pencil image')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(sketch, cmap='gray')
    plt.title('sketch low-res')
    plt.axis('off')

    plt.suptitle('Resolution {}, kernel size:{}'.format(resolution, kernel_sz))
    plt.show()


if __name__ == '__main__':
    parse = argparse.ArgumentParser()

    parse.add_argument('--voc', type=str, default='F:/IMAGE_DATABASE/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/',
                       help='The absolute path of VOC dataset')
    parse.add_argument('--img', type=str, default='', help='name of image file, no .jpg or .png needed')
    parse.add_argument('--kernel', type=int, default=15, help='Gaussian blur kernel size in image 2 pencil step')
    parse.add_argument('--gamma', type=float, default=2.0, help='gamma transfer in pencilize process')
    parse.add_argument('--cuda', type=bool, default=False, help='use CUDA or not when doing deep learning')
    parse.add_argument('--model_skt', type=str, default='sketch_simplification/model_gan.t7', help='The path to the "sketch simplification" pretrained network')
    parse.add_argument('--vec_num', type=int, default=1, help='the number of vector form sketch data')

    args = parse.parse_args()
    args.img = '2007_000033'
    args.model_skt = 'sketch_simplification/model_gan.t7'

    # validation
    assert os.path.exists(args.voc), 'ERROR: VOC path is invalid!'
    assert args.voc[-1] == '/', 'The voc path should end up with "/"'
    assert args.kernel >= 3, 'kernel size is too small'
    assert args.model_skt[-2:] == 't7', 'The format of simplification network is incorrect'

    img, pencil = image2pencil(args.voc, args.img, args.kernel, args.gamma)
    sketch = pencil2sketch(pencil, args.model_skt, args.cuda)
    sketch2vector('data/', 'sketch.png', args.vec_num, model_base_dir='virtual_sketching/model/')

    canvas_draw('data/seq_data/sketch_0.npz', 32, 128)
    cv.imshow('', sketch)
    cv.waitKey(0)


