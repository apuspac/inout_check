import glob
import ic_module as ic
import os.path as os

dirname = "hantei"
files = glob.glob(dirname + "/*")
cn1 = 0
cn2 = 0

for imgname in files :
    kind = ic.TestProcess(imgname)

    if kind == 0:
        cn2 += 1
    cn1 += 1
    print(kind)

print("正答率:" + str(cn2*1.0/cn1) + "")