# inout_check
卒業研究で、制作した顔認証を使った在室管理システムです。

- 顔検出:  
Haar-like特徴を使ったcascade分類器を使う手法を用いた。
- 顔認証  
Kerasを使用して 顔画像を学習、少ない人数なら判別できる。
- 環境:  
RaspberryPiを使用。同じネットワークに接続することで，webからシステムを操作することができる。  
データベースはpostgreSQLを使用。
- UI:  
DjangoからWebから操作可能にした

現在動くかは 確認してません。