#%%
import h5py 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray
import scipy

#%%
#get the data from a folder
train_f = h5py.File('./catH5/train_catvnoncat.h5')
test_f = h5py.File('./catH5/test_catvnoncat.h5')
# %%
#get data from files
train_list_classes = train_f['list_classes'].value
train_set_x_origin = train_f['train_set_x'].value
train_set_y_origin = train_f['train_set_y'].value

test_list_classes = test_f['list_classes'].value
test_set_x_origin = test_f['test_set_x'].value
test_set_y_origin = test_f['test_set_y'].value

# %%
#preprocessing train_test data 
#first to flatten and standardize train_test data
train_set_x = train_set_x_origin.reshape(train_set_x_origin.shape[0], -1).T / 255
train_set_y = train_set_y_origin.reshape(1, train_set_y_origin.shape[0])
test_set_x = test_set_x_origin.reshape(test_set_x_origin.shape[0],-1).T/255
test_set_y = test_set_y_origin.reshape(1, test_set_y_origin.shape[0])

# %%
#initialize weights and b to zeros
def initialize_wb(nx):
    w = np.zeros((nx, 1))
    b = 0
    return w, b

# %%
#calculate a
def calculate_a(w,x,b):
    z = np.dot(w.T, x) + b
    result = sigmoid(z)
    return result

#define function sigmoid(z)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
  
# %%
#calculate current cost function 
def forward_propagation(a, y):
    loss = -y*np.log(a) - (1-y)*np.log(1-a)
    cost = np.sum(loss)/y.shape[1]
    return cost

# %%
#calculate backward_propagation(a,y)
def backward_propagation(a, x, y):
    dw_sum = np.dot(x, (a - y).T)
    dw = dw_sum / y.shape[1]
    db = np.sum(a - y)/y.shape[1]
    return{'dw': dw,
           'db': db}
#%% 
# combine the forward propagation and the backward propagation
def propagation(train_set_x, train_set_y, w, b):
    a = calculate_a(w, train_set_x, b)
    cost = forward_propagation(a,train_set_y)
    result = backward_propagation(a,train_set_x,train_set_y)
    dw = result['dw']
    db = result['db']
    return {
        'a': a,
        'cost': cost,
        'dw': dw,
        'db': db
    }
# %%
#calculate the optimization process by updating w and b 
def optimization(train_set_x, train_set_y, w, b, learning_rate, loop_number, print_cost = False):
    costs = []
    for i in range(loop_number):
        result = propagation(train_set_x,train_set_y,w,b)
        cost = result['cost']
        dw = result['dw']
        db = result['db']
        a = result['a']
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i%100 == 0:
            costs.append(cost)
        if print_cost and i%100 == 0:
            print('Cost after iteration %i:%f' %(i, cost))
    params = {'w':w,
              'b': b}
    grads = {'dw':dw,
             'db': db}
    return params, grads, costs
#%%    
#calculate prediction process 
def prediction(w, b, test_set_x):
    m = test_set_x.shape[1]
    y = np.zeros((1,m))
    a = calculate_a(w,test_set_x, b)
    for i in range(a.shape[1]):
        if a[0][i] > 0.5:
            y[0][i] = 1
        else:
            y[0][i] = 0
    return y 
#%%
#def model 
def model(train_set_x, train_set_y, test_set_x, test_set_y, learning_rate = 0.005, loops = 2000, print_cost=False):
    nx = train_set_x.shape[0]
    w,b = initialize_wb(nx)
    params, grads, costs = optimization(train_set_x,train_set_y,w,b,learning_rate ,loops,print_cost)
    w = params['w']
    b = params['b']
    m = train_set_x.shape[1]
    predict_y_test = prediction(w,b,test_set_x)
    predict_y_train = prediction(w, b, train_set_x)
    predict_errors_test = np.mean(np.abs(predict_y_test - test_set_y))
    predict_errors_train = np.mean(np.abs(predict_y_train - train_set_y))
    predict_errors = {'test_errors': predict_errors_test,
                      'train_errors': predict_errors_train}
    optimization_results = {'costs': costs,
                           'grads': grads,
                           'params': params}
    return predict_errors, optimization_results

#%%
#main program
predict_errors, optimization_results = model(train_set_x,train_set_y,test_set_x, test_set_y, learning_rate = 0.005, loops = 2000, print_cost = False)
#%% print accuracy data 
predict_accuracy_train = (1 - predict_errors['train_errors']) * 100
predict_accuracy_test = (1 - predict_errors['test_errors']) * 100
print(f'The accuracy for train data is {predict_accuracy_train}%')
print(f'The accuracy for test data is {predict_accuracy_test}%') 

# %% print the coss curve 
costs = optimization_results['costs']
x_values = np.arange(0,2000,100)
plt.plot(x_values,costs)
plt.ylabel('Cost')
plt.xlabel('Number of Iterations')

# %%
num_px = train_set_x_origin.shape[1]
image = Image.open('non-cat1.jpeg')
image = image.resize((num_px,num_px))

image_np_array = asarray(image)
image_np_array = image_np_array/255
image_flatten = image_np_array.reshape(-1)
predict_data_x = image_flatten.reshape((image_flatten.shape[0],1))

w = optimization_results['params']['w']
b = optimization_results['params']['b']

predict_result = prediction(w,b,predict_data_x)
print(predict_result)

#%%