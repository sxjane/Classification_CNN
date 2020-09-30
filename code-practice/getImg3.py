
# %%
import zlib
import io
import numpy as np
from PIL import Image

content = open("6.npyz", "rb").read()
content = zlib.decompress(content)
arr = np.load(io.BytesIO(content))


# %%
type(arr)
arr.dtype

# %%
first = arr[:,:,0]
second = arr[:,:,1]
third = arr[:,:,2]

# %%
firstImg = Image.fromarray(first)
secondImg = Image.fromarray(second)
thirdImg = Image.fromarray(third)

#%%
reFirst = first.reshape((1024,1024,1))
reFirst.shape
#%%
reSecond = second.reshape((1024,1024,1))
reSecond.shape
#%%
reThird = third.reshape((1024,1024,1))
reThird.shape

# %%
result = np.concatenate((reFirst, reSecond, reThird), axis=2)
result.shape
# %%
imgRe = first + second + third
imgRe
Image.fromarray(imgRe)

# %%
a = np.zeros((1024,1024), dtype = np.uint8)
a

# %%
m = arr.shape[2]
for i in range(m):
    a = a + arr[:,:,i]
a
# %%
Image.fromarray(a)

# %%
