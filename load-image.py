#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 配列変数を扱うのに必要なライブラリ
import numpy as np
# 画像の入出力につかうライブラリたち
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import pylab

# "PetImages/Cat/0.jpg" というファイルを開いて画像をRGB形式で読み込みます
img=Image.open("PetImages/Cat/0.jpg").convert('RGB')

# 読み込まれた画像をnumpyの配列変数形式に変換します
# 3次元配列の3つの添字は、順に色(channel)、y座標、x座標を表します
img =np.asarray(img).astype(np.float32).transpose(2, 0, 1)

# 画像データの中身や、配列の寸法を表示します
print img
print img.shape
print img[0].shape

# ファイル名とnumpy形式の画像を受け取って、画像ファイルを書き出す関数を定義します
def write_image(fn, img):
    pylab.rcParams['figure.figsize'] = (6.4,6.4)
    pylab.clf()
    pylab.imshow((img.transpose(1,2,0))/255.0)
    pylab.savefig(fn)


# まずは読み込んだとおりの猫画像を'test-normal-cat.jpg'書き出してみます
write_image('test-normal-cat.jpg', img)

# 元々の画像の赤色を2倍にした画像を作ります
red_img = 1.0 * img
red_img[0, :, :] =  np.minimum(255, 2.0 * img[0] )
write_image('test-red-cat.jpg', red_img)

# 元々の画像の緑色を2倍にした画像を作ります
green_img = 1.0 * img
green_img[1, :, :] =  np.minimum(255, 2.0 * img[1] )
write_image('test-green-cat.jpg', green_img)

# 左上が暗い画像をつくります
halfdark_img = 1.0 * img
n_color, n_y, n_x = img.shape
for c in range(n_color):
    for y in range(n_y):
        for x in range(n_x):
            if x < 300 and y < 200:
                halfdark_img[c, y, x] = halfdark_img[c, y, x] /2
write_image('test-halfdark-cat.jpg', halfdark_img)

# 猫のプライバシーに配慮した画像を作ります
privacy_img = 1.0 * img
privacy_img[:, 150:170, 150:220] = 0
write_image('test-privacy-cat.jpg', privacy_img)
