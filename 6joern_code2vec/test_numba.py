import time
import pandas as pd
from numba import jit


# @jit
def time_com(i):
    cum = 0
    for test in range(i):
        for ind in range(i):
            cum += (test * ind) % 3


if __name__ == '__main__':
    t1 = time.time()
    df = pd.DataFrame()
    for i in range(500):
        time_com(i)
    t2 = time.time()
    print(t2-t1)