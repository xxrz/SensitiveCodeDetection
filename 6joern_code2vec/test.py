import random

import numpy as np

from string import digits

#生成-0.25-0.25之间的数
# W=(np.random.random(100)-0.5)/2
# # print(W)
#
# #随机打乱
# a = [1,2,3]
# a = np.array(a)
# # x = np.random.permutation(a)
# x = np.random.shuffle(a)
# print(a)

# s = 'abc123def456ghi789zero0'
# remove_digits = str.maketrans('', '', digits)
# res = s.translate(remove_digits)
# print(res)

b = "a1a2a3s3d4f5fg6h"
filter(lambda x: x.isalpha(), b)
print(b)