import sys
import os
import cv2
import glob
import shutil

from statistics import mode
from .capture import capture_image
from .authentication import ic_module as ic

IMAGE_DATA_PATH = "./face_login/app/face_image_data/"

def preprocess():
    """学習準備
    画像データを読み込んでnpyファイルを作る
    """
    i = 0
    for npyFileName in ic.npyDataNames:
        dirname = IMAGE_DATA_PATH + ic.dirNames[i]
        print(dirname)

        ic.PreProcess(dirname, npyFileName, var_amount=5)
        i+=1

def learning():
    """学習
    preprocessで生成したnpyファイルを使って学習モデルと重みファイルを作る。
    """
    ic.Learning(tsnum=45, nb_epoch=50, batch_size=8, learn_schedule=0.9)

def test_Authentication(imgname):
    """一枚用顔画像認証
    テスト用の一枚だけのラベル判定

    args:
        imgname(str): testに入っているファイルの名前
    """
    test_image_data = IMAGE_DATA_PATH + "test/" + imgname
    ic.TestProcess(test_image_data)

def file_Authentication(dirname):
    """認証部分

    フォルダ内の画像を読み込んで、判定。その後ラベルを保存して一番多かったラベルを判定結果として出す
    args:
        dirname(str): 画像ファイルのディレクトリパス
    """
    imgfile_path = glob.glob(IMAGE_DATA_PATH + dirname + "/*")
    print(imgfile_path)
    img_id_list = []
    for imgname in imgfile_path:
        img_id_list.append(ic.TestProcess(imgname))
        print(imgname)

    print(img_id_list)
    img_id = mode(img_id_list)

    print(img_id)
    return(img_id)


##ここからボタン化かな
def prepare_anthentication():
    """prepare_anthentication
    画像nampy化と学習
    """
    preprocess()
    learning()

def authentication():
    """判定
    顔画像を5枚程度取って判別する。
    """
    capture_image.face_capture("test", 5)
    id_num = file_Authentication("test")
    #shutil.rmtree(IMAGE_DATA_PATH + "test/")    #テスト内ファイル削除
    print(id_num)
    return str(id_num)

#capture("4")
#repare_anthentication()
#test_Authentication("ok.jpg")
#authentication()

#os.path.exists("./learning_data/img1.npy")
