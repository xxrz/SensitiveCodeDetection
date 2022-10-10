import numpy as np
import datetime

data = np.arange(500000000)
print("begin:")
begin = datetime.datetime.now()
data = data * 2
end = datetime.datetime.now()
print((end-begin).microseconds)

# list = list(range(10000000))
# begin = datetime.datetime.now()
# list = list * 2
# end = datetime.datetime.now()
# print((end-begin).microseconds)

# print((data.tolist()))