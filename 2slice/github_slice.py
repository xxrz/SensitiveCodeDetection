import os

def getFunc(spath,tpath):
    if not os.path.exists(tpath):
        os.makedirs(tpath)
    files = os.listdir(spath)
    for file in files:
        sourcePath = spath + '//' + file
        targetPath = tpath + '//' + file.split(".")[0] + ".c"
        try:
            with open(sourcePath, 'r', encoding='UTF-8') as f:
                for line in f:
                    if '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^' not in line:
                        with open(targetPath, 'a', encoding='UTF-8') as f2:
                            f2.write(line)
                    else:
                        with open(targetPath, 'a', encoding='UTF-8') as f2:
                            f2.write("---------------------------------\n")
                        break
        except:
            print(sourcePath)

if __name__ == '__main__':
    # spath = r'D:\XRZ\补充数据实验\data\CWE-704\github\str分类\备份数据\good'
    # tpath = r'D:\XRZ\Data\joern\github\CWE\CWE-704\good'
    # getFunc(spath,tpath)
    spath = r'D:\XRZ\Data\sard\C\切片\CWE-119\bad'
    tpath = r'D:\XRZ\Data\joern\github\CWE\CWE-704\good'
    getFunc(spath, tpath)
    print('over')
