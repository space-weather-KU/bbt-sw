#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 太陽磁場画像を取得するサンプルプログラムです

# 必要なライブラリのimport文です
import datetime, math,os, random,scipy.ndimage, StringIO, sys, urllib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import pylab
from astropy.io import fits
from observational_data import *

import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import serializers

# パラメータ群
if len(sys.argv) < 2:
    print "usage: {} data-folder-path".format(sys.argv[0])
    exit(2)
set_data_path(sys.argv[1])

learning_batchsize = 10
learning_image_size = 256

initial_learn_count = 10000
predict_count = 365 * 24 * 4
learn_per_step = 1
global current_hour
current_hour = 365 * 24

workdir = 'hmi-to-goes-forecast'
try:
    os.mkdir(workdir)
except Exception as e:
    pass

model_filename = workdir + '/model.save'
state_filename = workdir + '/state.save'


# ConvolutionとBatchNormalizationをセットでやるchainです
class ConvBN(chainer.Chain):
    def __init__ (self,in_nc, out_nc, ksize=3, stride=2):
        super(ConvBN, self).__init__(
            f = L.Convolution2D(in_nc, out_nc, ksize=ksize, stride=stride,
                                wscale=0.02*math.sqrt(ksize*ksize*in_nc)),
            g = L.BatchNormalization(out_nc)
        )
    def __call__ (self, x):
        return self.g(self.f(x))


# 256^2サイズの画像から１つの値を予測するニューラルネットワークです
class Predictor(chainer.Chain):
    def __init__(self):
        super(Predictor, self).__init__(
            l1=ConvBN(  1, 16,3,stride=2),
            l2=ConvBN( 16, 32,3,stride=2),
            l3=ConvBN( 32, 64,3,stride=2),
            l4=ConvBN( 64,128,3,stride=2),
            l5=ConvBN(128,256,3,stride=2),
            l6=ConvBN(256,512,3,stride=2),
            l7=ConvBN(512,1024,3,stride=2),
            l9=L.Linear(1024,1)
        )

    def __call__(self, x):
        h = F.leaky_relu(self.l1(x))
        h = F.leaky_relu(self.l2(h))
        h = F.leaky_relu(self.l3(h))
        h = F.leaky_relu(self.l4(h))
        h = F.leaky_relu(self.l5(h))
        h = F.leaky_relu(self.l6(h))
        h = F.leaky_relu(self.l7(h))
        return F.exp(self.l9(h)-13.0)

################################################################
# メインプログラム開始
################################################################

# ニューラルネットワークによるモデルと、モデルの最適化機を作ります
model = Predictor()
optimizer = optimizers.Adam(alpha = 0.0001)
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
    print('saved.')




class InOutPair:
    def visualize(self, fn_base):
        hmi_fn = fn_base + '-hmi.png'
        goes_fn = fn_base + '-goes.png'

        img = np.arctan(self.hmi_img / 300.0)
        img3 = np.zeros((1024,1024,3), dtype=np.float32)
        img3[:,:,0] = np.minimum(1,np.maximum(-1,img))/2+0.5
        img3[:,:,1] = np.minimum(1,np.maximum(-1,img))/2+0.5
        img3[:,:,2] = np.minimum(1,np.maximum(-1,img))/2+0.5
        pylab.rcParams['figure.figsize'] = (6.4,6.4)
        pylab.clf()
        pylab.gca().set_title('SDO/HMI Line-of-sight at {}(TAI)'.format(self.time))
        pylab.imshow(img3)
        pylab.savefig(hmi_fn)
        pylab.close('all')

        pylab.rcParams['figure.figsize'] = (6.4,4.8)
        pylab.gca().set_yscale('log')
        days    = mdates.DayLocator()  # every day
        daysFmt = mdates.DateFormatter('%Y-%m-%d')
        hours   = mdates.HourLocator()
        pylab.gca().xaxis.set_major_locator(days)
        pylab.gca().xaxis.set_major_formatter(daysFmt)
        pylab.gca().xaxis.set_minor_locator(hours)
        pylab.gca().grid()
        pylab.gcf().autofmt_xdate()
        pylab.plot(self.goes_lightcurve_t, self.goes_lightcurve_y, 'b', zorder=300)

        predict_t = [self.time, self.time+datetime.timedelta(days=1)]
        predict_y = [self.goes_max_predict, self.goes_max_predict]
        observe_t = [self.time, self.time+datetime.timedelta(days=1)]
        observe_y = [self.goes_max, self.goes_max]

        pylab.plot(predict_t, predict_y, color=(1,0,0), lw=1, zorder = 200,marker='o',linestyle='--')
        pylab.plot(observe_t, observe_y, color=(1,0.66,0.66), lw=2, zorder = 100)

        pylab.gca().set_xlabel('International Atomic Time')
        pylab.gca().set_ylabel(u'GOES Long[1-8A] Xray Flux')

        pylab.savefig(goes_fn)
        pylab.close('all')

def visualize_log():
    logs = []
    predicts = []
    observes = []
    with open(workdir + '/log.txt','r') as fp:
        for l in iter(fp.readline, ''):
            ws = l.split()
            t = datetime.datetime.strptime(ws[0],"%Y-%m-%dT%H:%M")
            p = float(ws[1])
            o = float(ws[2])
            predicts.append(p)
            observes.append(o)

    pylab.rcParams['figure.figsize'] = (6.4,6.4)
    pylab.gca().set_xscale('log')
    pylab.gca().set_yscale('log')
    pylab.scatter(predicts,observes, color=(1,0,0), zorder = 100,marker='.',s=1)
    pylab.scatter(predicts[-100:],observes[-100:], color=(1,0,0), zorder = 200,marker='o',s=5)
    pylab.scatter(predicts[-10:],observes[-10:], color=(1,0,0), zorder = 300,marker='o',s=10)

    pylab.gca().set_xlabel('prediction')
    pylab.gca().set_ylabel('observation')
    pylab.gca().grid()
    pylab.savefig(workdir+'/log.png')
    pylab.close('all')


def predict(learn_mode = True):
    global current_hour
    batch = []
    batchsize = learning_batchsize if learn_mode else 1

    while len(batch) < batchsize:
        if learn_mode:
            # 学習モードの場合
            # 2011年初頭からcurrent_hour - 24のあいだでランダムな時刻tを生成します
            step = random.randrange(current_hour-24)
            t = datetime.datetime(2011,1,1,0,0) + datetime.timedelta(hours=step)
        else:
            # 予測モードの場合
            # 対象時刻はcurrent_hourです
            t = datetime.datetime(2011,1,1,0,0) + datetime.timedelta(hours=current_hour)

        # 時刻tのHMI画像の取得を試みます
        img = get_hmi_image(t)
        if img is None:
            if learn_mode:
                continue # だめだったら別のtを試す
            else:
                return None # あきらめる

        # 時刻、画像、GOESライトカーブなどの情報を持ったInOutPairを作ります。
        p = InOutPair()
        p.time = t
        p.hmi_img = img
        p.goes_max = max(1e-8, get_goes_max(t, datetime.timedelta(days=1), data_path = data_path))

        p.goes_lightcurve_t = []
        p.goes_lightcurve_y = []
        t2 = t - datetime.timedelta(days=1)
        while t2 < t + datetime.timedelta(days=2):
            x2 = get_goes_flux(t2, data_path = data_path)
            if x2 is not None:
                p.goes_lightcurve_t.append(t2)
                p.goes_lightcurve_y.append(x2)
            t2 += datetime.timedelta(minutes=1)

        batch.append(p)

    # ニューラルネットワークに対する入力を作ります
    input = np.ndarray((batchsize,1,learning_image_size,learning_image_size), dtype=np.float32)
    for i in range(batchsize):
        h,w = batch[i].hmi_img.shape
        input[i,:,:,:] = scipy.ndimage.interpolation.zoom(batch[i].hmi_img,
                                                          (learning_image_size/float(h),learning_image_size/float(w)))

    input_v = chainer.Variable(input)
    predict_v = model(input_v)
    predict = predict_v.data
    for i in range(batchsize):
        batch[i].goes_max_predict = predict[i,0]

    if learn_mode:
        # 学習モードの場合、正解とのずれを学習します
        observe = np.ndarray((batchsize,1), dtype=np.float32)
        for i in range(batchsize):
            observe[i] = batch[i].goes_max
            observe_v = chainer.Variable(observe)

        def square_norm(x,y):
            return F.sum((F.log(x)-F.log(y))**2)/batchsize

        optimizer.update(square_norm, predict_v, observe_v)

    else:
        # 予報モードの場合、予報結果と正解を記録します。
        with open(workdir + '/prediction-log.txt','a') as fp:
            for p in batch:
                fp.write(' '.join([p.time.strftime("%Y-%m-%dT%H:%M"),str(p.goes_max_predict),str(p.goes_max),"\n"]))


# まず、最初の1年間で練習します
for i in range(initial_learn_count):
    print "learning: ", i, "/", initial_learn_count
    try:
        predict(learn_mode = True)
    except Exception as e:
        print str(e.message)

    if i % 100 == 0:
        save()
        #visualize_log()


#時間を1時間づつ進めながら、予報実験をしていきます。
for t in range(predict_count):
    print "predicting: ", t, "/", predict_count
    try:
        predict(learn_mode = False)
    except Exception as e:
        print str(e.message)

    for i in range(learn_per_step):
        try:
            predict(learn_mode = True)
        except Exception as e:
            print str(e.message)
    current_hour += 1
