#! -*- coding: utf-8 -*-
import glob
import numpy as np
import matplotlib.pyplot as plt
import pydotplus
import graphviz



from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.preprocessing.image import random_rotation, random_shift, random_zoom
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.models import Sequential
from keras.models import model_from_json
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
#from tensorflow.python.keras.utils.vis_utils import plot_model


#処理共通のパラメータ
npyDataNames = ["img1.npy", "img2.npy", "img3.npy"]
dirNames =["img1", "img2", "img3"]
ClassNames = ["1", "2", "3"]
hw = {"height":32, "width":32}        # リストではなく辞書型 中かっこで囲む
LEARNING_DATA_PATH = "./face_login/app/learning_data/"

# class LearningParameters:
#     npyDataNames = ["img1.npy", "img2.npy", "img3.npy"]
#     dirNames =["img1", "img2", "img3"]
#     ClassNames = ["1", "2", "3"]
#     hw = {"height":32, "width":32}


def PreProcess(dirname, npyFileName, var_amount=3):
    """画像データのnumpy化

    画像ファイルをリサイズしてで読み込み、numpy配列に変換後、リストに格納。
    npyファイルとして保存する。

    args:
        dirname(string): 画像ファイルが入っているディレクトリのpath
        npyFileName(string) : 保存するnpy配列のファイルの名前
        var_amount(int) : 画像ファイルかさまし数の変更

    """
    num = 0
    arrlist = []
    # フォルダ内のファイルパスをリスト化
    files = glob.glob(dirname + "/*")
    #files = glob.glob(dirname + "/*.jpeg")

    # 処理部分：ランダムに回転させることで画像データのかさましを行っている。
    for imgfile in files:
        img = load_img(imgfile, target_size=(hw["height"], hw["width"]))    # 32*32で画像ファイルの読み込み
        array = img_to_array(img) / 255                                     # 画像ファイルのnumpy化
        arrlist.append(array)                 # numpy型データをリストに追加
        for i in range(var_amount-1):
            arr2 = array
            arr2 = random_rotation(arr2, rg=360)
            arrlist.append(arr2)              # ランダム回転かさましnumpy型データをリストに追加
        num += 1

    nplist = np.array(arrlist)
    np.save(LEARNING_DATA_PATH + npyFileName, nplist)   #保存
    print(">> " + dirname + "から" + str(num) + "個のファイル読み込み成功")


def BuildCNN(ipshape=(32, 32, 3), num_classes=3):
    """CNNモデル構築

    画像判定に用いるCNNモデルを構築する。ついでに学習modelも作成してくれている。

    args:
        ipshape(tuple): (32, 32, 3)は32×32で　画像数3つという意味
        num_classes(int): 最終的に分類する数。クラス分けする数

    returns:
        model: 構築したmodelを返す。
    """
    model = Sequential()

    #3*3フィルタ 畳み込み処理を24回行う
    #padding=same 周りは0
    model.add(Conv2D(24, 3, padding='same', input_shape=ipshape))
    model.add(Activation('relu'))

    #3*3フィルタの畳み込み処理を48回
    # maxpooling2D poolsizeの最大値を出力する。
    #画像データを2×2の小領域に分割し、その中の最大値を出力
    model.add(Conv2D(48, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5)) #過学習を抑えるために入力の50%を0に置き換え なぜ？？

    #↑2つの処理が96回
    model.add(Conv2D(96, 3, padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(96, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    #Flatten() dense(128)により要素128の一次元配列にする。
    #denseは全結合層 1次元にしないといけないため。
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    #出力の個数を読み込んだフォルダの数(=画像の種類)
    #
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # 構築
    # 最適化関数 adam
    # 損失関数 categorical_crossentropy
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    # 学習モデル図の作成
    #pydotplusでgraphvizを使う必要あり
    #plot_model(model, to_file='model.png')
    plot_model(model, show_shapes=True, expand_nested = True)
    model.summary()

    return model


def plot_history(history):
    """精度グラフ作成

    精度のグラフを作成し、表示する

    args:
        history: kerasのhistoryを渡す。
    """
    # 精度の履歴をプロット
    plt.plot(history.history['accuracy'],"o-",label="accuracy")
    plt.plot(history.history['val_accuracy'],"o-",label="val_acc")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    plt.show()

    # 損失の履歴をプロット
    plt.plot(history.history['loss'],"o-",label="loss",)
    plt.plot(history.history['val_loss'],"o-",label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.show()


#epoch = 50 変更
def Learning(tsnum=30, nb_epoch=50, batch_size=8, learn_schedule=0.9):
    """Learning

    モデルを使って学習をする。
    npyファイルを読み込み、学習用とテスト用に分ける。
    学習モデルはjsonファイル,
    重みはhdf5にて保存する。

    args:
        tsnum(int): 精度確認用画像枚数
        nb_epoch(int): 学習の繰り返し回数
        batch_size(int): 入力データを平均化する大きさ
        learn_schedule(float):重みを就職しやすくするための学習率の倍数？


    """
    #TRAIN_listは学習用, TEST_listはテスト用
    X_TRAIN_list = []; Y_TRAIN_list = []; X_TEST_list = []; Y_TEST_list = []
    class_number = 0  #分類番号

    #入力データの画像と教師データの分類番号を関連付ける
    # npyFileName と 分類番号を紐づけさせる。
    for npyFileName in npyDataNames :
        data = np.load(LEARNING_DATA_PATH + npyFileName)          # 画像のnumpyデータを読み込み
        trnum = data.shape[0] - tsnum
        X_TRAIN_list += [data[i] for i in range(trnum)]          # 画像データ
        Y_TRAIN_list += [class_number] * trnum                         # 分類番号
        X_TEST_list  += [data[i] for i in range(trnum, trnum+tsnum)]          # 学習しない画像データ
        Y_TEST_list  += [class_number] * tsnum;                                     # 学習しない分類番号
        class_number += 1

    X_TRAIN = np.array(X_TRAIN_list + X_TEST_list)    # 連結
    Y_TRAIN = np.array(Y_TRAIN_list + Y_TEST_list)    # 連結
    print(">> 学習サンプル数 : ", X_TRAIN.shape)
    y_train = np_utils.to_categorical(Y_TRAIN, class_number)    # 自然数をベクトルに変換(1,2,3)-> [1, 0, 0][0,1,0]
    valrate = tsnum * class_number * 1.0 / X_TRAIN.shape[0] #データ全体のうちどれくらいの割合を精度確認用にするか


    # 学習率の変更関数
    class Schedule(object):
        def __init__(self, init=0.001):      # 初期値定義
            self.init = init
        def __call__(self, epoch):           # 現在値計算
            lr = self.init
            for i in range(1, epoch+1):
                lr *= learn_schedule

            print(lr)
            return lr

    def get_schedule_func(init):
        return Schedule(init)

    #学習率変更関数
    lrs = LearningRateScheduler(get_schedule_func(0.001))
    #val_loss が最も小さくなるたびに、重みを保存する関数
    mcp = ModelCheckpoint(filepath=LEARNING_DATA_PATH +'best.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    #BuildCNN 構築の学習モデル
    model = BuildCNN(ipshape=(X_TRAIN.shape[1], X_TRAIN.shape[2], X_TRAIN.shape[3]), num_classes=class_number)
    #early_stopping = EarlyStopping(patience=0, verbose=1)

    print(">> 学習開始")
    hist = model.fit(X_TRAIN, y_train,  #データの指定
                     batch_size=batch_size,     #入力データの平均化の大きさ
                     verbose=1,
                     epochs=nb_epoch,
                     validation_split=valrate,  #精度確認用データの割合
                     callbacks=[lrs, mcp])

    #json形式で保存 分類名も保存
    json_string = model.to_json()
    json_string += '##########' + str(ClassNames)
    open(LEARNING_DATA_PATH + "model.json", 'w').write(json_string)
    model.save_weights(LEARNING_DATA_PATH + 'last.hdf5')

    plot_history(hist)


def TestProcess(imgname):
    """実験用

    画像を読み込み、学習結果から判定する。

    args:
        imgname: 画像ファイル名
    """


    modelname_text = open(LEARNING_DATA_PATH + "model.json").read()
    json_strings = modelname_text.split('##########')
    textlist = json_strings[1].replace("[", "").replace("]", "").replace("\'", "").split()
    model = model_from_json(json_strings[0])            # model読み込み
    model.load_weights(LEARNING_DATA_PATH + "last.hdf5")                     # best.hdf5 で損失最小のパラメータを使用

    #画像の読み込み
    img = load_img(imgname, target_size=(hw["height"], hw["width"]))
    TEST = img_to_array(img) / 255      #numpy行列に変換

    pred = model.predict(np.array([TEST]), batch_size=1, verbose=0)
    print(">> result:\n" + str(pred))
    print(">> label:" + textlist[np.argmax(pred)].replace(",", "") + "")

    class_id = int(textlist[np.argmax(pred)].replace(",", ""))

    return class_id
    #ファイル一括処理
    #return np.argmax(pred)