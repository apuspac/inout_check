"""this is face.py test program"""
import sys
import os
from datetime import datetime
import cv2
import time

#CASCADE_PATH = "cvmain/haarcascades/haarcascade_frontalface_default.xml"
CASCADE_PATH = "./face_login/app/capture/cvmain/haarcascades/haarcascade_frontalface_alt2.xml"
IMAGE_DATA_PATH = "./face_login/app/face_image_data/"

DEVICE_ID = 0
IMAGE_COUNT = 0

def face_detect(image):
    """
    画像を読み込んで、顔認識をさせ、検出した顔画像の座標をfacerectに帰す。
    """
    #グレースケール変換
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #分類器を読み込む
    cascade = cv2.CascadeClassifier(CASCADE_PATH)

    # detectMultiScale
    # 入力画像中から異なるサイズの物体を検出。矩形のリストとして返す
    # scaleFactor 各画像における縮小量を表す
    # minneighbor 物体候補となる矩形は、最低でもこの数だけの近傍矩形を含む
    # minsize 物体とする最小サイズ これより小さいのは検出しても無視

    #検出した顔画像の座標を格納する
    #[左上 x, 左上y, 右下x, 右下y] かな？
    facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.2, minNeighbors=2, minSize=(10, 10))

    return facerect


def save_faceImage(image_path, facerect, base=256, range={"width":0, "height":0}, save_path='img'):
    """
    顔画像のみを保存する
    """
    global IMAGE_COUNT
    if type(image_path) is str: # 画像ファイルのパスで受け取ったとき
        image = cv2.imread(image_path)

        #画像ファイル名の取り出し
        #/で区切ったpathの最後のやつ取り出し ./image/gazou.ppm のgazou.ppm
        #.で区切った最初のやつ取り出し gazou.ppmの gazou
        image_path = image_path.split("/")[-1].split(".")[0]
    else: #画像を受け取ったとき
        image = image_path
        #現在の時間を取得してimagepathに
        image_path = datetime.now().strftime("%Y-%m-%d-%H%M%S")

    #ディレクトリの作成
    if len(facerect) > 0:
        # save_path = save_path
        if not os.path.exists(save_path): #pathの存在確認 なかったら作る
            os.mkdir(os.path.join(save_path))

    for i , rect in enumerate(facerect): #オブジェクトに加えてindexを取得できる 1 gohan, 2 pan 的な
        #ここの処理分からん
        if rect[2] < base:
            continue

        x = rect[0] - range["width"]
        y = rect[1] - range["height"]
        width = rect[2] - range["width"]
        height = rect[3] - range["height"]

        # a:s aからsまでの意味 顔切り出し
        dst = image[y:y+height, x:x+width]

        # 画像を保存
        new_image_path = save_path + '/' + image_path + "_" + str(i) + ".jpg"
        cv2.imwrite(new_image_path, dst)
        print(new_image_path + "is clip and saved")
        time.sleep(1)
        IMAGE_COUNT += 1

def camera_facedetect(save_path, capture_count):
    """
    カメラで画像を取得し、顔認識・顔だけ保存を呼び出す
    """
    global IMAGE_COUNT
    cap = cv2.VideoCapture(DEVICE_ID)
    end_flag, frame = cap.read()

    while (IMAGE_COUNT < capture_count):
        if cv2.waitKey(1) == 27: #escape
            break

        # 顔の検出と保存
        image = frame
        face_list = face_detect(image)

        print(len(face_list))

        for(x, y, w, h) in face_list:
            cv2.rectangle(image, (x,y), (x+w, y+h), (255, 0, 0), 2)


        cv2.imshow("", image)
        save_faceImage(image, face_list, base=64, save_path=save_path)
        ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()
    IMAGE_COUNT = 0

def face_capture(save_img_dirname, capture_count):
    """顔画像取得

    取得後は指定したディレクトリに保存してくれる

    args:
        save_img_dirname(str): 保存するフォルダのパスを書く。face_image_data以下のパスを記述
        capture_count(int): 保存する画像の枚数
    """
    #データ保存用のフォルダ なかったら作るようのラムダ関数
    mkdirExceptExist = lambda path: "" if os.path.exists(path) else os.mkdir(path)
    mkdirExceptExist("./face_login/app/face_image_data")
    save_dir_path = IMAGE_DATA_PATH + save_img_dirname + "/"
    mkdirExceptExist(save_dir_path)

    camera_facedetect(save_dir_path, capture_count)

if __name__ == "__main__":
    ARGS = sys.argv     #コマンドライン引数を格納したリストの取得
    ARGC = len(ARGS)    #引数の取得

    # 引数指定 img/
    if(ARGC != 2):
        print("保存path指定plz")
        quit()

    user_id = ARGS[1]
    face_capture(user_id)

# if __name__ == "__main__":
#     ARGS = sys.argv     #コマンドライン引数を格納したリストの取得
#     ARGC = len(ARGS)    #引数の取得

#     # 引数指定 img/
#     if(ARGC != 2):
#         print("保存path指定plz")
#         quit()

#     SAVE_PATH = ARGS[1]
#     main(SAVE_PATH)