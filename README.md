
# このチュートリアル以外の資料

Pythonの練習 -> https://github.com/space-weather-KU/bbt-sw/blob/master/python-cheatsheet.md
参考文献リスト -> https://github.com/space-weather-KU/bbt-sw/blob/master/%E8%B3%87%E6%96%99.md
Pythonの教科書は色々ありますし、Web上の資料も充実していますが、１つ挙げるなら[柴田淳(著)「みんなのPython」 第3版] (http://amzn.to/1QlWqGR) がおすすめです。
    

# このチュートリアルの使い方


## 準備

以下の説明で、`$`で始まる行は、キーボードから入力するコマンドを表します。
先頭の`$`は除いて、のこりを入力してください。


### プログラミング言語Pythonとライブラリのインストール

Python2/Python3という選択肢がありますが、今回参考にするHMI Science Nuggets がPython2で書かれているので当面Python2を使おうと思います。

Anacondaがあれば[Python3の環境を作ってAnaconda内から切り替えられます](http://conda.pydata.org/docs/py2or3.html#create-python-2-or-3-environments)

- Windows
    - [Anacondaをインストールする](https://www.continuum.io/downloads) のがおすすめです。
    - [Windows用　参考記事](http://qiita.com/y__sama/items/5b62d31cb7e6ed50f02c#windows%E3%81%AE%E5%A0%B4%E5%90%88)
- Mac OSX
    - homebrewを使うのが便利です。 http://qiita.com/zaburo/items/fbdaf6c04151671407db
    - Anacondaを入れる場合は若干注意が必要です。こちら [Mac OSX用　参考記事](http://qiita.com/oct_itmt/items/2d066801a7464a676994)
- Unix系
    - 各ディストリビューションのパッケージマネージャ等を使ってください。
    - LinuxにもAnacondaがおすすめです。　[参考記事](http://qiita.com/y__sama/items/5b62d31cb7e6ed50f02c#linux%E3%81%AE%E5%A0%B4%E5%90%88)

### 太陽観測データの扱い

後期は、太陽観測データを本格的に扱う必要がありますので、どのような方法が良いか調査してきました。皆様に以下の方法で行けそうか試してもらえればと思います
。

HMI Science Nuggetsの記事を参考にします。
- [50. Analyzing SDO/HMI Data Using Python](http://hmi.stanford.edu/hminuggets/?p=1428)

ここに[S1] .. [S5]としてリンクされている先は、Pythonの実行可能なメモ帳「Jupyter」のソースコードです。


- ここのS3の記事 [making movies of the Sun](http://nbviewer.jupyter.org/github/mbobra/calculating-spaceweather-keywords/blob/master/movie.ipynb) が実行できて、太陽の活動領域のムービーを作成するところまで行けるか試してください。みなさまの環境で躓くところをおしえてください。

Jupyterは、AnacondaやOS付属のコマンドラインから次のコマンドを入力するとインストールできます。

```
$ pip install --user jupyter
```
上記のサイトから`movie.ipynb`をウンロードしてきてください。そして、`movie.ipynb`の存在するフォルダに
移動して
```
jupyter notebook
```
としてjupyterを起動してください。
うまくいったら、Analyzing SDO/HMI Data Using Pythonの他のスクリプトや、他のHMI Science Nuggetsも試してみてください。

- 動画生成に必要な`ffmpeg`ライブラリはここ https://anaconda.org/menpo/ffmpeg にあるように、Anacondaからインストールできます。
```
conda install -c menpo ffmpeg
```
と入力してください。

- `movie.ipynb`の最後まで行って動画は生成されるが生成した動画が再生できない場合、最後の行に`extra_args=['-vcodec','libx264']`オプションを追加するとうまくいく可能性があります。

```
ani.save(NOAA_ARS+'.mp4', savefig_kwargs=savefigdict, writer='ffmpeg_file', dpi=my_dpi,extra_args=['-vcodec','libx264'])
```

### Chainerと必要なライブラリのセットアップ
まずは、上記手順を参考に, python, pipをインストールしてください。そして、

```
$ pip install --user chainer
```

を実行して、chainer (1.8以降)をインストールして下さい。

他に、matplotlib, PILも必要なのでインストールします。
```
$ pip install --user matplotlib
$ pip install --user PIL
```

Windowsでは、PILの代わりにpillowを使うのが良いようです。
```
$ pip install --user matplotlib
$ pip install --user pillow
```



### スクリプト一式の入手
gitをインストールし、コマンド

```
$ git clone https://github.com/space-weather-KU/bbt-sw.git
```

を実行するか、あるいはこのページの上の方にある"Download ZIP"ボタンからファイル一式を手に入れてください。

次に、犬猫画像を用意します。
後述のASIRRA Cat and Dog image corpus
http://research.microsoft.com/en-us/projects/asirra/corpus.aspx
から 885MB の猫犬画像データセットをダウンロードするか、あるいはUSBメモリから受けとって、
さきほどgithubから手に入れたフォルダ（`cat-or-dog.py`などがあるところ）に、PetImages/Cat　フォルダおよび
 PetImages/Dogフォルダがあるようにしてください。





## 機械学習の実行

- プログラミング言語 python の処理系
- pythonで書かれた深層学習ライブラリ chainer
- 学習につかうための犬と猫の画像のデータセット

が準備できました。

この状態で、
```
$ ./cat-or-dog.py
```
を実行すると、犬と猫の画像にもとづく学習が始まります。学習結果は`model.save`および `state.save`という２つのファイルにセーブされます。

十分学習させるまで数時間待つか、あるいは次のコマンドで予め学習済みのセーブデータを移動してきて、
```
$ mv learned/* .
```

次のようにして、ある画像が猫か犬かを判定出来ます。
```
$ ./cat-or-dog.py PetImages/Cat/0.jpg
Load model from model.save state.save
PetImages/Cat/0.jpg  is a  cat
```

自分で撮った写真や、インターネット上から探してきた画像でも試してみてください。

```
~/hub/bbt-sw (master)$ ./cat-or-dog.py tama.jpg
Load model from model.save state.save
tama.jpg  is a  cat
```

## 宇宙天気予報との関係は？

このプログラムを応用して、たとえば「ある時刻の太陽画像」から「２４時間後に起こるフレアのクラス(X,M,C)」を予測したり、「７２時間後までに起こる最大のフレアの規模(W/m^2)」につなげていきます。

他にも、「画像から何かの分類・数値を予測する」という様々な用途に応用できます。


# 気になる論文

## [NEW] 動画予測・生成系

- PredNet https://arxiv.org/abs/1605.08104
- Generating Videos with Scene Dynamics　http://web.mit.edu/vondrick/tinyvideo/
- WaveNet (音声)　https://deepmind.com/blog/wavenet-generative-model-raw-audio/
- Temporal Generative Adversarial Nets https://arxiv.org/pdf/1611.06624v1.pdf


- 自然言語の文章により指定された物体だけを、画像内から選ぶ http://www.eccv2016.org/files/posters/S-1A-07.pdf

## 画像認識系

- ASIRRA Cat and Dog image corpus
http://research.microsoft.com/en-us/projects/asirra/corpus.aspx

- Googleの「猫ニューロン」論文
http://static.googleusercontent.com/media/research.google.com/ja//archive/unsupervised_icml2012.pdf

- ホテルの部屋の画像を作る(画像生成系の始祖)
http://arxiv.org/pdf/1511.06434v2.pdf

- DCGAN Face Generator
https://mattya.github.io/chainer-DCGAN/

- 画像から英文を作る
https://cs.stanford.edu/people/karpathy/cvpr2015.pdf

- 英文から画像を作る
http://arxiv.org/pdf/1511.02793v2.pdf

## 時系列系

- LSTM による自動翻訳
https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf

- Playing Atari Game with Deep Reinforcement Learning
http://arxiv.org/pdf/1312.5602.pdf

- Alpha Go
http://www.nature.com/nature/journal/v529/n7587/pdf/nature16961.pdf

## 授業で使用するデータ

### 2016-06-27

GOES観測データ　１時間おきにサンプルしたもの
https://github.com/space-weather-KU/bbt-sw-samples/blob/master/goes-data-1hr.txt?raw=true

