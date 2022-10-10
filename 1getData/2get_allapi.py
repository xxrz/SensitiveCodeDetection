import os
import json

#功能：得到所有commits中的api，并去重
#修改参数：需要修改commits的path路径，以及最后保存文件的名称

def get(rootpath):
    files = os.listdir(rootpath)
    for file in files:
        path = rootpath + '//' + file
        try:
            with open(path, 'r', encoding='utf-8') as f:
                t = json.load(f)
                items = t['items']
                for item in items:
                    with open('allapi.txt', 'a', encoding='utf-8') as file:
                        file.write(item['url'] + "\n")
        except:
            print(path)

#去重
def uniq(path):
    writePath = 'new.txt'
    lines_seen = set()
    with open(path, 'r', encoding='utf-8') as r:
        for line in r:
            if line not in lines_seen:
                with open(writePath, 'a+', encoding='utf-8') as w:
                    w.write(line)
                    lines_seen.add(line)

#得到所有api+去重
def synthesize(rootpath):
    files = os.listdir(rootpath)
    set=[]
    for file in files:
        path = rootpath + '//' + file
        try:
            with open(path, 'r', encoding='utf-8') as f:
                t = json.load(f)
                items = t['items']
                for item in items:
                    #修改需要保存文件的名称
                    with open(rootpath+'-allapi.txt', 'a', encoding='utf-8') as file:
                        if item['url'] not in set:
                            file.write(item['url'] + "\n")
                            set.append(item['url'])
        except:
            print(path)


if __name__ == '__main__':
    # path = r'2014'
    # get(path)
    # uniq('allapi.txt')
    # os.remove('allapi.txt')
    # os.renames('new.txt','allapi.txt')
    # print("over")
    path = r'data/improperAuthentication/improperAuthentication'
    synthesize(path)
    print("over")

