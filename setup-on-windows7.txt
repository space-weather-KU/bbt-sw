1)Windows環境へのPythonインストール
	http://qiita.com/maisuto/items/404e5803372a44419d60
	　参照
2)Python WindowsにPIPをインストール
	http://typea.info/tips/wiki.cgi?page=Python+Windows%A4%CBPIP%A4%F2%A5%A4%A5%F3%A5%B9%A5%C8%A1%BC%A5%EB#p0
	　参照
3)Chainerをインストール
	http://soysoftware.sakura.ne.jp/archives/561
	　参照
　1. Microsoft Visual C++ Compiler for Python 2.7をインストール
　2. C:\Python27\Scripts で pip install --user chainer
4)matplotlibをインストール
　C:\Python27\Scripts で pip install --user matplotlib
（この状態でcat-or-dog.pyを走らせると、「PILが入っていないと言われる）
↓
5)（PILの代わりに）Pillowをダウンロード
　C:\Python27\Scripts で pip install --user Pillow

後はプログラム・画像の準備などしてcat-or-dog.pyをダブルクリック
