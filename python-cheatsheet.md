# Pythonをインストールせずに利用する方法

オンラインの学習サイトを使いましょう。これらの学習サイトの中には、ブラウザ上でPythonを実行してくれるものがあるので、自分のパソコンにPytohnをインストールしなくても利用できます。

Codecademy https://www.codecademy.com/learn/python がPythonの基礎から１つづつ教えてくれるのでおすすめです。
Checkio https://checkio.org/



# Pythonインタプリタの利用

このゼミでは、基本的に、Pythonをシェル(コンピュータに命令を文字で入力するインターフェイス)から使います。
Windows使いの方はコマンドプロンプトやPowershell, Mac使いの方はTerminalなどを使ってください。

Pythonがインストールされている環境では、シェルに`python`と入力するとPythonの対話型環境が起動します。ここでは、Pythonのプログラムを入力して、Pythonに答えを計算してもらうことができます。

```
$ python
Python 2.7.9 (default, Mar  1 2015, 12:57:24)
[GCC 4.9.2] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> 6*7
42
>>> 1+1 == 2
True
>>> range(1,10)
[1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> sum(range(1,10))
45
```

# Pythonスクリプトの作成と実行

Pythonのプログラムの拡張子は`.py`です。次の内容を
好きなファイル名、たとえば`'first.py'`で保存し、実行してみてください。

```
#!/usr/bin/env python
print sum(range(1,101))
```

実行するには2つの方法があります。

1. `python first.py` として、Pythonに実行してもらう。
1. `chmod 755 first.py`などとしてこのファイルに実行権限を与え、 `./first.py`で実行する。


```
import numpy
img = numpy.zeros((5,3), dtype=np.float32)
print img
```
