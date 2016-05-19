# おことわり

この資料は [Python 2.7 Quick Reference Sheet](http://www.astro.up.pt/~sousasag/Python_For_Astronomers/Python_qr.pdf) を参考にしました。

# Pythonをインストールせずに利用する方法

オンラインの学習サイトを使いましょう。これらの学習サイトの中には、ブラウザ上でPythonを実行してくれるものがあるので、自分のパソコンにPytohnをインストールしなくても利用できます。

Codecademy https://www.codecademy.com/learn/python がPythonの基礎から１つづつ教えてくれるのでおすすめです。
Checkio https://checkio.org/ はもうちょっと発展的な課題のプログラムを作っていく感じです。

これらの学習サイトは課金するとより多くの演習問題が解禁されたりしますが、個人的にはこれらのサイトに課金する段階に達したなら、かえってPythonの教科書を買ったほうがずっと使えて良いかとは思います。

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

# Help

`dir()`関数をつかうと任意のオブジェクトのメソッドの一覧を見られます。たとえば、`x`という変数名の数のリストを作り、`dir()`を見てみます。

```
>>> x = [1,4,9]
>>> x
[1,4,9]
>>> dir(x)
['__add__', ... , 'append', 'count', ... ]
```

`append`というメソッドがあるようですね。これの`help()`を見てみます。

```
>>> help(x.append)

Help on built-in function append:

append(...)
    L.append(object) -- append object to end
(END)
```

`append`は、「リストの末尾にオブジェクトを追加する」とあります。試してみましょう。

```
>>> x.append(16)
>>> x
[1, 4, 9, 16]
>>>
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

## Import文

pythonでは、モジュール(他の人が作ったプログラム集)を利用するのに`import`文を使います。

```
import numpy
img = numpy.zeros((5,3), dtype=np.float32)
print img
```

次のようにすると、任意の名前でモジュールをインポートできます。

```
import numpy as np
img = np.zeros((5,3), dtype=np.float32)
print img
```
