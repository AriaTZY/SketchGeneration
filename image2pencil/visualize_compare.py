# this .py file visualize the world images, pencil images and sketch images for comparison
import cv2 as cv
import numpy
import matplotlib.pylab as plt
import os
from PIL import Image
import numpy as np
import PyQt5

# ====================
# parameters config
# ====================

root = 'E:\Postgraduate\sketch_simplification-master\data/'
root = 'C:/Users/tan\Desktop\data/'
image_path = root + 'image/'
pencil_path = root + 'input/'
sketch_path = root + 'output/'
sketch2_path = root + 'high res/'

start_idx = 10  # from which picture to show

# ====================
# start visualization
# ====================


def load_image(name, mode=1):
    if mode == 0:
        img = Image.open(name)
    else:
        img = cv.imread(name, 1)
        if img is not None:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


def generate_canvas(img, pen, skt, save=False):
    size = [1, 4]
    # ===================
    # show: first row
    # ===================
    plt.figure(figsize=(16, 4))
    plt.subplot(size[0], size[1], 1)
    plt.imshow(img)
    plt.title('world image')
    plt.axis('off')

    plt.subplot(size[0], size[1], 2)
    plt.imshow(pen)
    plt.title('pencil image')
    plt.axis('off')

    plt.subplot(size[0], size[1], 3)
    plt.imshow(skt)
    plt.title('sketch low-res')
    plt.axis('off')

    plt.subplot(size[0], size[1], 4)
    # Superimpose two pics
    plt.imshow(superimpose(img, skt))
    plt.title('Superimpose sketch lines')
    plt.axis('off')
    if save:
        plt.savefig('pyUI/visualization.png')
        plt.close()


# Superimpose two pics
def superimpose(img, skt):
    if len(skt.shape) == 2:
        skt = cv.cvtColor(skt, cv.COLOR_GRAY2RGB)
    img = cv.resize(img, (skt.shape[1], skt.shape[0]))
    superpose_img = skt * 0.7 + img * 0.3
    superpose_img = np.array(superpose_img, np.uint8)
    return superpose_img


if __name__ == '__main__':

    item_name_list = os.listdir(image_path)
    total_num = len(item_name_list)
    show_num = total_num - start_idx

    for i in range(show_num):
        # read images in
        name = item_name_list[i+start_idx]
        img = load_image(image_path + name)
        pen = load_image(pencil_path + name)
        skt = load_image(sketch_path + name)
        skt2 = load_image(sketch2_path + name)

        generate_canvas(img, pen, skt)

        plt.show()

