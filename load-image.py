#!/usr/bin/env python

import sys

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import pylab


with open("PetImages/Cat/0.jpg", 'r') as fp:
    img=Image.open(fp).convert('RGB')

img =np.asarray(img).astype(np.float32).transpose(2, 0, 1)
print img
print img.shape
print img[0].shape

def write_image(fn, img):
    pylab.rcParams['figure.figsize'] = (6.4,6.4)
    pylab.clf()
    pylab.imshow((img.transpose(1,2,0))/255.0)
    pylab.savefig(fn)



write_image('test-normal-cat.jpg', img)

red_img = 1.0 * img
red_img[0, :, :] =  np.minimum(255, 2.0 * img[0] )
write_image('test-red-cat.jpg', red_img)

green_img = 1.0 * img
green_img[1, :, :] =  np.minimum(255, 2.0 * img[1] )
write_image('test-green-cat.jpg', green_img)

privacy_img = 1.0 * img
privacy_img[:, 150:170, 150:220] = 0
write_image('test-privacy-cat.jpg', privacy_img)
