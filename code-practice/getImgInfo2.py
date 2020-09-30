#%%
from PIL import Image
import numpy as np 

im = Image.open("./black.png", );
im

#%%
imToNp = np.asarray(im);
imToNp
imToNp.shape

# %%
greyIm = im.convert(mode='L')
greyIm

# %%
greyImToNp = np.asarray(greyIm);
greyImToNp
greyImToNp.shape

# %%
greyImToNp.max()

# %%
greyImToNp.dtype

# %%
greyImToNp.min()


# %%
sp = greyImToNp * 255
sp
finalImage = Image.fromarray(sp)
finalImage

# %%
