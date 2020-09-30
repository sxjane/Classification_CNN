#%%
from PIL import Image
import numpy as np
im = Image.open("/Users/sg1/Desktop/1.png")
im
# %%
a=np.asarray(im)

# %%
type(a)

# %%
a.shape


# %%
a[0][0][0]



# %%
a[0]

# %%
a[0].shape

# %%
a[1,1,1]

# %%
Image.fromarray(a[:,:,0])

# %%
b = a.copy()
b[:,:,0] = 0
b[:,:,2] = 0
Image.fromarray(b)

# %%

# %%
