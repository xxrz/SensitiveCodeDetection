from gensim.models import KeyedVectors as word2vec
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
import numpy as np
import pandas as pd
from string import digits
import re
import os
import time
#用不了（被修饰的函数中不能调用第三方库）
from numba import jit

#利用了模糊匹配解决了joern和code2vec中不一致的token的问题 但是这样信息不全并且速度很慢 进一步的优化中

import warnings

from sklearn.random_projection import GaussianRandomProjection

warnings.filterwarnings("ignore")

operators3 = {'<<=', '>>='}
operators2 = {
    '->', '++', '--',
    '!~', '<<', '>>', '<=', '>=',
    '==', '!=', '&&', '||', '+=',
    '-=', '*=', '/=', '%=', '&=', '^=', '|='
    }
operators1 = {
    '(', ')', '[', ']', '.',
    '+', '-', '*', '&', '/',
    '%', '<', '>', '^', '|',
    '=', ',', '?', ':' , ';',
    '{', '}'
    }


def tokenize(line):
    tmp, w = [], []
    i = 0
    while i < len(line):
        # Ignore spaces and combine previously collected chars to form words
        if line[i] == ' ':
            tmp.append(''.join(w))
            tmp.append(line[i])
            w = []
            i += 1
        # Check operators and append to final list
        elif line[i:i + 3] in operators3:
            tmp.append(''.join(w))
            tmp.append(line[i:i + 3])
            w = []
            i += 3
        elif line[i:i + 2] in operators2:
            tmp.append(''.join(w))
            tmp.append(line[i:i + 2])
            w = []
            i += 2
        elif line[i] in operators1:
            tmp.append(''.join(w))
            tmp.append(line[i])
            w = []
            i += 1
        # Character appended to word list
        else:
            w.append(line[i])
            i += 1
    # Filter out irrelevant strings
    res = list(filter(lambda c: c != '', tmp))
    return list(filter(lambda c: c != ' ', res))

#token id map
def tokensMap(file,word):
    id = list(pd.read_csv(filepath_or_buffer=file)["id"].values)
    token =  list(pd.read_csv(filepath_or_buffer=file)["token"].values)
    for i in range(len(token)):
        if token[i]==word:
            return id[i]
    else:
        return None

#节点之间模糊匹配
def tokensfuzzMap(file,word):
    id = list(pd.read_csv(filepath_or_buffer=file)["id"].values)
    token = list(pd.read_csv(filepath_or_buffer=file)["token"].values)
    #模糊，score_cutoff为阈值
    key = process.extractOne(word, token, scorer=fuzz.ratio, score_cutoff=55)

    if key:
        # res = model.most_similar(key[0])
        # print()
        for i in range(len(token)):
            # 注意token中可能出现的异常值null None 由于写入修改的内容在csv文件中太麻烦，暂时不做异常处理
            if token[i] == key[0]:
                print(word,key[0])
                return id[i]
    else:
        return None

#相比之下效果很差
def test(file,word):
    id = list(pd.read_csv(filepath_or_buffer=file)["id"].values)
    token = list(pd.read_csv(filepath_or_buffer=file)["token"].values)
    for i in range(len(token)):
        if fuzz.partial_ratio(word,token[i])>50:
            print(word,token[i])
            return id[i]
    else:
        return None

#获取每个函数文件的节点关系的开始行
def getfileLine(path):
    with open(path,'r',encoding='utf-8') as f:
        linenumber = 0
        for readline in f.readlines():
            # print(re.match("-*",readline).span()[1],linenumber)
            if re.match("-*",readline).span()[1]!=0:
                # print(linenumber)
                break
            else:
                linenumber = linenumber + 1
        return linenumber + 1

#得到每个函数文件的token值+tag值（为保证得到的是相同函数的token和tag,所以放在一起处理）
def getWords_Tag(path):
    words = []
    linenumber = getfileLine(path)
    with open(path, 'r', encoding='utf-8') as f:
        readlines = f.readlines()
        for readline in readlines[linenumber:-2]:
            linewords = []
            # '\([^(.*,)]*'
            t = re.search('(?<=,).*', readline)
            if t:
                a = t.group().strip()
                b = re.sub(r'[()]', " ", a)
                b = tokenize(b)
                # b = re.sub(r'_',"", b)
                # b = b.lower()
                # if len(b) >8:
                #     remove_digits = str.maketrans('', '', digits)
                #     b = b.translate(remove_digits)
                # b = b.split(' ')
                # for i in b:
                #     # if i!='' and i!="'" and i!='if' and i!='malloc' and i!='sizeof':
                #         if i not in linewords:
                #             linewords.append(i)
            words.append(b)
        tag = readlines[-1]
    return words,tag

if __name__ == '__main__':
    # vectors_text_path = 'tokens.txt'
    # model = word2vec.load_word2vec_format(vectors_text_path, binary=False)
    # res = model.most_similar("data")  #字典里的词查询上下文
    # a = model.wv.vocab  #可查看字典里所有词

    # 生成的所有函数的路径
    # target_path_dir = r"D:\XRZ\Ubuntu\data\result\CWE-191"
    # target_paths = os.listdir(target_path_dir)
    #
    # matrixs = []
    # tags = []
    #
    # for target_path in target_paths:
    #     target_path = os.path.join(target_path_dir,target_path)
    #     words,tag = getWords_Tag(target_path)
    #
    #     matrix = np.zeros((100, 100))
    #     for i in range(min(len(words), 100)):
    #         wordline = words[i]
    #         tag = 0
    #         vector = np.zeros((1, 100))
    #         for word in wordline:
    #             if word in model.wv.vocab.keys():
    #                 if tag == 0:
    #                     vector = model.wv[word]
    #                     # vector = vector[:50]
    #                     vector = np.array(vector)
    #                     tag = 1
    #                 else:
    #                     vector = np.row_stack((vector, model.wv[word]))
    #         if np.where(vector!=0)[0].shape[0]==0:
    #             print(wordline,i)
    #         vector = np.array(vector)
    #         if np.size(vector) > 100:  # 降维
    #             # vector=np.array(vector).sum(axis=0)
    #             vector = vector.T
    #             gauss_proj = GaussianRandomProjection(n_components=1)
    #             vector = gauss_proj.fit_transform(vector).T
    #         matrix[i] = vector
    #     tags.append(tag)
    #     print("")
    #
    matrix = np.zeros((100, 100))
    words, tag = getWords_Tag(r"testresult/bad/9b35d9ba-6e5f-11ea-9cd7-b46bfc404df8.c-FUN1.txt")
    # for i in range(min(len(words), 100)):
    #     wordline = words[i]
    #     try:
    #         vector = model.wv[wordline]  # 查词
    #     except:
    #         # vector.append(np.random.rand(100))
    #         print(i,wordline)
    #         continue
    #     vector = np.array(vector)
    #     if np.size(vector) > 100:  # 降维
    #         # vector=np.array(vector).sum(axis=0)
    #         vector = vector.T
    #         gauss_proj = GaussianRandomProjection(n_components=1)
    #         vector = gauss_proj.fit_transform(vector).T
    #     matrix[i] = vector
    # print("over")


