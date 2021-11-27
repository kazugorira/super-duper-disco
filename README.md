# super-duper-disco

「SIGNATE Student Cup 2021秋：オペレーション最適化に向けたシェアサイクルの利用予測」で作ったモデルに手を加え「【SOTA】SIGNATE Student Cup 2021秋：オペレーション最適化に向けたシェアサイクルの利用予測」で検証したもの

City2とそれ以外のCityで分けたモデル、０時時点の自転車数を特徴量と加えたモデル、０時時点の自転車数を特徴量と加えなかったモデルでスタッキングした。

LB：3.7113888

City2とそれ以外では自転車数の変動が大きく異なっていたのとCity2が観光都市にあたるため分けて学習してみた。

参考文献
[【Python覚書】LightGBMで交差検証を実装してみる](https://potesara-tips.com/lightgbm-k-fold-cross-validation/)
