#%%
import numpy as np
import h5py
from PIL import Image
import matplotlib.pyplot as plt

# %%
train_file = h5py.File('/Users/sg1/Develop/PY/catH5/train_catvnoncat.h5')

# %%
train_list_classes = train_file['list_classes'].value
train_data_x = train_file['train_set_x'].value
train_data_y = train_file['train_set_y'].value

# %%
empty_data = np.zeros((1,64,64,3), dtype='uint8')

# %%
train_data_x_add = np.concatenate((train_data_x,empty_data),axis=0)
train_data_y_add = np.concatenate((train_data_y,[0]))

# %%
train_data_x_batches = np.reshape(train_data_x_add, (10,21,64,64,3))
train_data_y_batches = np.reshape(train_data_y_add, (10,21))
# %%

# %%
