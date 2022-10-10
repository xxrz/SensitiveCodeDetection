import os

rootpath = r'D:\XRZ\Data\sard\C\切片\CWE-191\g'
files = os.listdir(rootpath)
targetpath = r'good.txt'
for file in files:
    path = rootpath + '//' + file
    try:
        with open(path,'r',encoding='UTF-8') as f:
            for line in f:
                if '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^' not in line:
                    with open (targetpath,'a',encoding='UTF-8') as f2:
                        f2.write(line)
                else:
                    with open(targetpath, 'a',encoding='UTF-8') as f2:
                        f2.write("---------------------------------\n")
                    break
    except:
        print(path)
print('over')
