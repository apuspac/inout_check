import ic_module as ic
import os.path as op

i = 0
for npyFileName in ic.npyDataNames :
    # ディレクトリ名入力
    dirname = "../face_image_data/" + ic.dirNames[i]
    print(dirname)

    # 関数実行
    ic.PreProcess(dirname, npyFileName, var_amount=3)
    i += 1

# i = 0
# for filename in ic.FileNames :
#     # ディレクトリ名入力
#     while True :
#         dirname = input(">>「" + ic.ClassNames[i] + "」の画像のあるディレクトリ ： ")
#         if op.isdir(dirname) :
#             break
#         print(">> そのディレクトリは存在しません！")

#     # 関数実行
#     ic.PreProcess(dirname, filename, var_amount=3)
#     i += 1