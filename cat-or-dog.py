#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob,random, sys, subprocess, os.path

import matplotlib
matplotlib.use('Agg')
import pylab

import numpy as np
from PIL import Image

import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import serializers

# batchsizeは、一度に学習する画像の枚数です
batchsize = 10
# 元画像から、このサイズの領域を取り出して学習します
imgsize = 64

model_filename = 'model.save'
state_filename = 'state.save'


def read_command(cmd):
    stdout, stderr = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    return stdout

catfns = glob.glob('PetImages/Cat/*')
dogfns = glob.glob('PetImages/Dog/*')


# 猫と犬の画像を分類するニューラルネットワークです。
class CatDog(chainer.Chain):
    def __init__(self):
        super(CatDog, self).__init__(
            l1=L.Convolution2D(3,64,3,stride=2),
            l2=L.Convolution2D(64,128,3,stride=2),
            l3=L.Convolution2D(128,256,3,stride=2),
            l4=L.Linear(256*7*7,2)
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        return F.reshape(self.l4(h3), (x.data.shape[0],2))


# fns が None なら、教師データのなかからランダムに画像を読み込みます。
# fns が None でなければ、指定されたファイル名の画像を読み込みます。
def load_image(fns = None):
    if fns is not None:
        my_batchsize = len(fns)
    else:
        my_batchsize = batchsize

    ret = np.zeros((my_batchsize, 3, imgsize, imgsize), dtype=np.float32)
    ret_ans = np.zeros((my_batchsize), dtype=np.int32)


    for j in range(my_batchsize):
        # 正解ラベルをランダムに選ぶ
        ans = np.random.randint(2)
        img = None
        if fns is not None:
            # fnsで指定されたファイル名の画像を読み込む
            img=Image.open(fns[j]).convert('RGB')
            ans=-1
        else:
            while True:
                try:
                    # 正解ラベルに従って猫、犬のいずれかの画像を読み込む
                    if ans==0:
                        fn = random.choice(catfns)
                    else:
                        fn = random.choice(dogfns)

                    with open(fn,'r') as fp:
                        img=Image.open(fp).convert('RGB')
                    break
                except:
                    continue

        # ランダムな回転と拡大縮小を加える
        img=img.rotate(np.random.random()*20.0-10.0, Image.BICUBIC)
        w,h=img.size
        scale = 80.0/min(w,h)*(1.0+0.2*np.random.random())
        img=img.resize((int(w*scale),int(h*scale)),Image.BICUBIC)
        img = np.asarray(img).astype(np.float32).transpose(2, 0, 1)

        # 画像の中央付近をランダムに切り出す
        oy = (img.shape[1]-imgsize)/2
        ox = (img.shape[2]-imgsize)/2
        oy=oy/2+np.random.randint(oy)
        ox=ox/2+np.random.randint(ox)

        # 1/2の確率で、左右を反転させる
        if np.random.randint(2)==0:
            img[:,:,:] = img[:,:,::-1]

        ret[j,:,:,:] = (img[:,oy:oy+imgsize,ox:ox+imgsize]-128.0)/128.0
        ret_ans[j] = ans
    return (chainer.Variable(ret), chainer.Variable(ret_ans))

# 与えられた画像をファイルに書き出す関数です
def write_image(fn, img, ans):
    pylab.rcParams['figure.figsize'] = (6.4,6.4)
    pylab.clf()
    pylab.imshow((img.transpose(1,2,0) + 1)/2)
    if ans == 0:
        pylab.title("cat")
    else:
        pylab.title("dog")
    pylab.savefig(fn)


################################################################
# メインプログラム開始
################################################################


# ニューラルネットワークによるモデルと、モデルの最適化機を作ります
model = L.Classifier(CatDog())
optimizer = optimizers.Adam()
optimizer.setup(model)

# セーブファイルが存在する場合は、セーブファイルから状態を読み込みます
if os.path.exists(model_filename) and os.path.exists(state_filename):
    print 'Load model from', model_filename, state_filename
    serializers.load_npz(model_filename, model)
    serializers.load_npz(state_filename, optimizer)

# 現在までの学習状態をファイルに書き出す関数です。
def save():
    print('save the model')
    serializers.save_npz(model_filename, model)
    print('save the optimizer')
    serializers.save_npz(state_filename, optimizer)

# 機械学習を行う関数です
def learn():
    e = 0
    while True:
        e+=1
        print e,
        imgs, anss = load_image()
        optimizer.update(model, imgs, anss)
        print model.loss.data

        write_image('test.png', imgs.data[0], anss.data[0])

        if e & (e-1) == 0 or e%1000 == 0:
            save()
    save()

# コマンドライン入力から与えられたファイル名の画像を、猫犬判定する関数です。
def test():
    fns = sys.argv[1:]
    for fn in fns:
        imgs, anss = load_image(11*[fn])
        model(imgs,anss)
        count_cat = 0
        count_dog = 0
        for c,d in model.y.data:
            if c>d:
                count_cat += 1
            else:
                count_dog += 1
        if count_cat > count_dog:
            result = "cat"
        else:
            result = "dog"

        print fn, " is a ", result

# コマンドライン引数からファイル名が与えられていれば、
# test()を呼びます。
# ファイル名が与えられていなければ学習を行います
if len(sys.argv) > 1:
    test()
else:
    learn()
