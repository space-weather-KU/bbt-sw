
# このチュートリアルの使い方


## 準備

以下の説明で、`$`で始まる行は、キーボードから入力するコマンドを表します。
先頭の`$`は除いて、のこりを入力してください。

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



python, pipをインストールしてください。そして、コマンド

```
$ pip install --user chainer
```

を実行して、chainer (1.8以降)をインストールして下さい。コンピュータ室の環境では、上記のコマンドで入るはずです。



## 機械学習の実行

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
