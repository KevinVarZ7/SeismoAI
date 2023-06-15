#!/usr/bin/env python
# coding: utf-8

# <center>
#    <img src="logounam.png"width="150">
# </center>
# 
# 
#   <font size="6"><b> <center> Universidad Nacional Autónoma de México     </b> <br> </font>
#   <font size="4"><b> <center> Posgrado en Ciencias de la Tierra </b><br> </font>
#   <font size="3"> <center> 1D CNN implementation for Regional Seismic Event detection in Paricutin Data </b> <br> </font>
#   <font size="3"><b> <center>@Author: MSc. Kevin Axel Vargas-Zamudio </b> <br> </font>
#   <font size="3"><b> <center>email: seismo.ai.kevvargas@gmail.com </b><br></font>

# # Observed data for 2022 Paricutin Seismic Swarm
# ## 1D CNN development for Regional Seismic Event Detection
# ### Convolutional Neural Network: Hyparameter tuning

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from scipy.io import loadmat,savemat

from os.path import dirname, join as pjoin
import sys
import copy as copy
import os
import time
import random


# In[46]:


def matfiles_retrieve_24(year,month,day):
    # Retrieving .mat files from /res directory for a single day
    path = sys.path[-1]

    mat_fname = []

    for i in range(0,24):
        if i <= 9:
            mat_fname.append(path + 'pickreg'+ year + '_'+ month + '_' + day +'_0' + str(i) + '.mat')
        else:
            mat_fname.append(path + 'pickreg'+year+'_'+ month + '_' + day + '_' + str(i) + '.mat')

    # .mat file list for a complete day (each .mat data file for 1 of 24 hours)
    mat_files = []
    for i in range(len(mat_fname)):
        mat_files.append(loadmat(mat_fname[i]))
        
    # Keys for data day dictionary
    data_day = list()
    for s in range(len(mat_fname[:])):
        data_day.append(mat_fname[s][75:-4])
        
    # Create a Dictionary that contains tp and ts variable length vectors for each hour
    # Sorted and repetitions removed
    TPTS_1day = {}
    hours = []
    for i in range(24):
        TP = mat_files[i]['tp'][0] ; TP = TP[~np.isnan(TP)] 
        TS = mat_files[i]['ts'][0] ; TS = TS[~np.isnan(TS)]
        TPTS_1day[data_day[i]+'_tp'] = np.unique(np.sort(TP))
        TPTS_1day[data_day[i]+'_ts'] = np.unique(np.sort(TS))
    
    keys = list(TPTS_1day.keys()) #'2022_09_21_03_tp'
    
    for j in range(0,len(keys),2):
        hours.append(int(keys[j][11:13]))

    return data_day, TPTS_1day, hours


# In[2]:


def retrieve_Xinput_Ytarget(years,months,day_30,day_31):
    # Retrieving .mat files from /res directory for a single day
    path = 'X_input_mat_regional/'
    mat_fname = []
    
    for year in years[0:1]:
        for month in months[7:13]:#[8:11]
            if  month == '01' or month == '03' or month == '05' or month == '07'             or month == '08' or month == '10' or month == '12':
                days = copy.copy(day_31)
            elif month == '04' or month == '06' or month == '09' or month == '11':
                days = copy.copy(day_30)

            for day in days:#[15:]     
                mat_fname.append(path + 'Xinput_'+ year + '_' + month + '_' + day + '.mat')
 
    # .mat file list for a complete day (each .mat data file for 1 of 24 hours)
    data = []
    for i in range(len(mat_fname)):
        data.append(loadmat(mat_fname[i]))

    # Keys for data day dictionary
    data_day = list()
    for s in range(len(data[:])):
        data_day.append(data[s])
    
    return data, data_day


# In[3]:


def retrieve_specific_Xinput_Ytarget(date):
    n = len(date)
    path = 'X_input_mat_regional/'
    mat_fname = []
    data = []
    
    for i in range(n):
        year = date[i][0:4]
        month = date[i][5:7]
        day = date[i][8:]
        print('Xinput_'+ year + '_' + month + '_' + day + '.mat')
    
    # Retrieving .mat files from /res directory for a single day
        mat_fname.append(path + 'Xinput_'+ year + '_' + month + '_' + day + '.mat')
    # .mat file list for a complete day (each .mat data file for 1 of 24 hours)
    
    #Xinput
    for i in range(len(mat_fname)):
        data.append(loadmat(mat_fname[i]))
        
    # Keys for data day dictionary
    data_day = list()
    for s in range(len(data[:])):
        data_day.append(data[s])  
    return data, data_day


# In[4]:


def month_day_str():
    month = []
    day1 = [] ; day2 = []

    for i in range(1,13):
        if i <= 9:
            month.append('0'+str(i))
        else:
            month.append(str(i))
    for i in range(1,31):
        if i <= 9:
            day1.append('0'+str(i))
        else:
            day1.append(str(i))
            
    day2 = copy.copy(day1)
    day2.append('31')
    
    return month, day1, day2


# In[5]:


months,day30,day31 = month_day_str()
year = '2022'
count = 0
for m in months[7:]:
    if m == '08' or m == '10' or m == '12':
        days = copy.copy(day31)
    else:
        days = copy.copy(day30)
    for d in days:
        count += 1
        print(f'Day #: {count}, {year}_{m}_{d} ')


# In[6]:


def manual_train_test_split(perc_train,perc_test,Xinput,ytarget):
    # Selecting random sample from global input dataset
    
    Ninput = X_input.shape[0]
    Nwin = X_input.shape[1]
    Ntrain = int(Ninput * perc_train)
    Ntest  = int(Ninput * perc_test )
    
    # Extracting random index vector
    X_idx = np.arange(0,Ninput,1)
    random.seed(7)
    train_idx = random.sample(list(X_idx),Ntrain)
    train_idx_sorted = sorted(train_idx)
    np.asarray(train_idx_sorted)
    
    test_idx = np.delete(X_idx,train_idx_sorted)
    test_idx = np.delete(test_idx,-1)
    print(len(train_idx_sorted),len(test_idx))
    
    # Excluding train samples from Total input samples = test samples
    Xtrain = X_input[train_idx_sorted]
    #Xtest= np.delete(X_input,train_idx_sorted,axis=0)
    Xtest = X_input[test_idx]
    
    # Target train and test vectors
    ytrain = Y_target[train_idx_sorted]
    ytest = Y_target[test_idx]
    ytrain = ytrain.astype(int)
    ytest = ytest.astype(int)
    
    Xtrain = Xtrain.reshape((Ntrain, Nwin, -1))
    Xtest = Xtest.reshape((Ntest, Nwin, -1))
    
    ytrain = np.reshape(ytrain,(-1,1))
    ytest  = np.reshape(ytest,(-1,1))

    #print(Ntrain,Ntest)
    #print(f'Xtrain Shape: {np.shape(Xtrain)} , ytrain Shape: {np.shape(ytrain)} \n')
    #print(f'Xtest Shape:  {np.shape(Xtest)}  , ytest Shape:  {np.shape(ytest)} \n')
    
    return Xtrain, ytrain, Xtest, ytest


# In[7]:


# Loading X_input noise
path = 'X_input_mat_regional/'
noise_mat = loadmat(path + 'Xinput_noise.mat')
x_noise = noise_mat['Xinput_noise']
y_noise = noise_mat['Ytarget_noise'].reshape(-1)

print(f'Noise X input data shape: {x_noise.shape} \nNoise y target label shape: {y_noise.shape}')


# In[8]:


# Reading X input and y target

#data = loadmat(path + 'Xinput_2022_09_03')
months , day_30, day_31 = month_day_str()
years = ['2022', '2023']

# loading X_input Signal
data,data_day = retrieve_Xinput_Ytarget(years,months,day_30,day_31)

Xinput = []
for i in range(len(data)):
    Xinput.append(data[i]['Xinput'])

ones = [2,19,22,49,51,56,57,58,85,107,136,137,150]

aux = 0
for i in range(len(data)):
    if i in ones:
        aux += len(data[i]['Xinput'][:-1])
    else:
        aux += len(data[i]['Xinput'])
        
print(f'Number of signals:{aux}')
# Adding the number of noise windows
aux += len(x_noise[:,0])

print(len(x_noise[:,0]),aux)
Nsamp_win = len(data[0]['Xinput'][0])
Nlabels   = len(data[0]['Ytarget'][0])

X_input = np.zeros((aux,Nsamp_win))
Y_target = np.zeros((aux))

aux = 0
for i in range(len(data)): # 153 days
    if i in ones:
        for j in range(len(data[i]['Xinput'][:-1])):   # variable lenght, number of 'good' events per day
            X_input[aux,:] = data[i]['Xinput'][j]
            Y_target[aux]  = data[i]['Ytarget'][0][j]
            aux += 1
    else:
        for j in range(len(data[i]['Xinput'])):
            X_input[aux,:] = data[i]['Xinput'][j]
            Y_target[aux]  = data[i]['Ytarget'][0][j]
            aux += 1

# #for j in range(len(noise)):
X_input[aux:,:] = x_noise[:,:]#x_noise[:50,:]
Y_target[aux:] = y_noise[:]

print(Nsamp_win,Nlabels)


# In[10]:


# ndays = len(data[:])
# for i in range(ndays):
#     print(i,data[i]['Xinput'].shape, data[i]['Xinput'][-1])


# In[11]:


# list_zero_days = [2,19,22,49,51,56,57,58,85,107,136,137,150]
# for i in list_zero_days:
#     print(i,data[i]['Xinput'].shape, data[i]['Xinput'], type( data[i]['Xinput']))


# In[12]:


#len(Y_target[1947:])


# In[9]:


# Calling manual function for splitting the Input Dataset
perc_train = 0.75
perc_test  = 0.25

Xtrain,ytrain,Xtest,ytest = manual_train_test_split(perc_train,perc_test,X_input,Y_target)

print(f'X,y train dataset shape: {np.shape(Xtrain)} , {np.shape(ytrain)}')
print(f'X,y test dataset shape:  {np.shape(Xtest)}  , {np.shape(ytest)}')


# In[10]:


ytest = ytest[:].reshape(-1).astype(int)
ytrain = ytrain[:].reshape(-1).astype(int)
print(ytrain)

# Counting number of labels for each class
print(np.unique(ytrain,return_counts=True))
print(np.unique(ytest,return_counts=True))


# # Neural Network Architecture, definition and parameters:
# - Convolutional layers:
# - Pooling layers:
# - Dropout layers:
# - Dense Layer
# - Activation function: Sigmoid, tanh, ReLu, Linear
# - Descent Gradient optimization method: ADAM
# - Loss Function: Binary Crossentropy 

# In[12]:


import tensorflow as tf
# 1D CNN architecture: Convolution, Dropout, Pooling, Dense
from tensorflow.keras.layers import Conv1D,Dense,Dropout,MaxPool1D,InputLayer,Flatten,GlobalAveragePooling1D
from tensorflow.keras import Input,optimizers
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler


# In[13]:


# Simple 1D Convolutional Neural Network for seismic time series classification, with arbitrary parameters
def CNN_simple_model(Xtrain,ytrain,Xtest,ytest):
    import time
    
    input_shape = tf.TensorShape(Xtrain[0].shape)
    units = 64
    nclass = 1
    pool_size = 2
    activation = 'sigmoid'
    filters = 16
    kernel_size = 3
    lr = 0.0001

    i = Input(shape=input_shape)
    x = Conv1D(filters=filters,kernel_size=kernel_size,activation=activation)(i) # Sigmoid for Classification
    x = MaxPool1D(pool_size = pool_size)(x)
    x = Flatten()(x)
    x = Dense(units,activation = activation)(x)
    x = Dense(nclass,activation='sigmoid')(x)

    model_cnn = Model(i,x)
    model_cnn.summary()

    # Training the convolutional neural network
    start_time = time.time()

    # Stablish the model compile parameters
    model_cnn.compile(loss='binary_crossentropy',optimizer=optimizers.Adam
                  (learning_rate=lr),metrics = 'accuracy')
    verbose = 0
    mod_cnn_fit = model_cnn.fit(Xtrain,ytrain,validation_data=(Xtest,ytest),epochs=100,verbose=verbose)
    score   = model_cnn.evaluate(Xtest,ytest,verbose=verbose)

    print("---CNN1D training and testing time execution: %s seconds ---" % (time.time() - start_time))
    
    return model_cnn,mod_cnn_fit,score,units,activation,kernel_size,lr,filters


# In[14]:


# Multiblock 1D Convolutional Neural Network for seismic time series classification, with optimized hyperparameters
def CNN_multiblock_model(Xtrain,ytrain,Xtest,ytest):
    import time
    
    input_shape = tf.TensorShape(Xtrain[0].shape)
    units = 64
    nclass = 1
    pool_size = 2
    activation = 'sigmoid'
    filters = 32
    kernel_size = 11
    lr = 0.0001
    
    i = Input(shape=input_shape)
    # First block ------
    x = Conv1D(filters=filters,kernel_size=kernel_size,activation=activation)(i) # Sigmoid for Classification
    x = MaxPool1D(pool_size = pool_size)(x)
    x = Dropout(0.25)(x)
    
    # Second Block -----
    x = Conv1D(filters = 2*filters,kernel_size=kernel_size,activation=activation)(x)
    x = MaxPool1D(pool_size = pool_size)(x)
    x = Dropout(0.25)(x)
    
    # third Block -----
    x = Conv1D(filters = 4*filters,kernel_size=kernel_size,activation=activation)(x)
    x = MaxPool1D(pool_size = pool_size)(x)
    x = Dropout(0.25)(x)
    
    # Flatten - Output layers
    x = Flatten()(x)
    x = Dense(units,activation = activation)(x)
    x = Dropout(0.25)(x)
    x = Dense(nclass,activation='sigmoid')(x)

    model_cnn = Model(i,x)
    model_cnn.summary()

    # Training the convolutional neural network
    start_time = time.time()

    # Stablish the model compile parameters
    model_cnn.compile(loss='binary_crossentropy',optimizer=optimizers.Adam
                  (learning_rate=lr),metrics = 'accuracy')
    verbose = 0
    mod_cnn_fit = model_cnn.fit(Xtrain,ytrain,validation_data=(Xtest,ytest),epochs=200,verbose=verbose)
    score   = model_cnn.evaluate(Xtest,ytest,verbose=verbose)
    
    # Saving model:
    model_cnn.save('CNNModels/CNN1D_3CPFrFDO')

    print("---CNN1D training and testing time execution: %s seconds ---" % (time.time() - start_time))
    
    return model_cnn,mod_cnn_fit,score,units,activation,kernel_size,lr,filters


# In[15]:


def plot_loss_acc_func_simple_parameter(mod_cnn,model_name,units,activation,lr,kernel_size,filters):    
    fig = plt.figure(figsize=(20,5))
    plt.suptitle(f'{model_name} Training: Loss and accuracy functions, Units: {units}, {activation}, Out: {activation}, Lr: {lr},    kernel: {kernel_size}, Filters: {filters}',fontsize=15,weight='bold')
    ax1 = plt.subplot(121)
    ax1.title.set_text('Loss')
    ax1.plot(mod_cnn.history['loss'],color='red',linewidth=2.5,marker = '.',label='train_loss')
    ax1.plot(mod_cnn.history['val_loss'],label='test_loss',linewidth=2.5,linestyle='-')
    ax1.set_xlabel('Epochs',fontsize=12,weight='bold')
    ax1.legend()
    ax1.grid(alpha=0.5)
    ax2 = plt.subplot(122)
    ax2.title.set_text('Accuracy')
    ax2.plot(mod_cnn.history['accuracy'],color='green',linewidth=2.5,marker='.',label='train_acc')
    ax2.plot(mod_cnn.history['val_accuracy'],label='test_acc',linewidth=2.5,linestyle='-')
    ax2.set_xlabel('Epochs',fontsize=12,weight='bold')
    ax2.legend()
    ax2.grid(alpha=0.5)
    fig.savefig(f'{model_name}_Loss_Accuracy_Adam_Units{str(units)}_{activation}_OPTIM.png'                ,bbox_inches='tight',pad_inches=0,dpi=100)


# In[16]:


# Matriz de confusión
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(model,cm,classes,normalize=False,
                           title='Confusion Matrix',cmap = plt.cm.Reds):
    if normalize:
        
        cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        print('matriz de confusión normalizada')
    else:
        print('Matriz de confusión sin normalización')
  
    print(cm)
    fig = plt.figure(figsize=(20,5))
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title+' '+model)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],fmt),
             horizontalalignment='center',
             color='white' if cm[i,j] > thresh else 'black')
        
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted label')
    #plt.show()
    plt.savefig(f'Confusion Matrix {model_name}.png',dpi=200,bbox_inches='tight',pad_inches=0.05)


# In[17]:


def model_test_evaluation(scores,model,Xtest,ytest,model_name):
    
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    # Test evaluation through Confusion matrix
    p_test = model.predict(Xtest)#.argmax(axis=0)

    print(f'Statistical Summary of Model predictions: \n     Mean = {np.mean(p_test)} \n Min = {p_test.min()} \n     Max = {p_test.max()}')

    plt.figure(figsize=(3,3))
    plt.hist(p_test)
    plt.grid(alpha=0.5)
    plt.title('Probability histogram for Test data')

    p_good = np.zeros((len(p_test[:])),dtype = int)

    for i in range(len(p_test[:])):
        if p_test[i]<=0.5:
            p_good[i] = 0
        else:
            p_good[i] = 1

    cm = confusion_matrix(ytest,p_good)
    plot_confusion_matrix(model_name,cm,list(range(2)))
    
    return p_good,cm


# In[18]:


def misclassified_data(model_name,ytest,p_good,cm):
    
    false_positives = cm[1,0] + cm[0,1]
    print('falsepos: ',false_positives)
    #rows = int(((np.ceil(np.sqrt(false_positives)) + 1)/3) - 1)
    rows = int(np.ceil(false_positives / 3))
    print('rows',rows)
    #cols = copy.copy(rows)
    
    mal_ind = np.where(p_good != ytest)[0]
    mal_ind = np.unique(mal_ind)

    mpl.rcParams['figure.figsize'] = [11,11]
    fig,ax = plt.subplots(rows,3,layout = 'constrained')

    aux = 0
    for i in range(rows):
        for j in range(3):
            if ytest[mal_ind[aux]] == 0:
                col = 'green'
            else:
                col = 'red'

            ax[i,j].plot(Xtest[mal_ind[aux]],color=col)
            ax[i,j].set_title('True label: %s , Predicted label: %s, Index: %s'                               %(ytest[mal_ind[aux]],p_good[mal_ind[aux]],mal_ind[aux]),fontsize=10)

            ax[i,j].tick_params(left = False, right = False , labelleft = False ,
                         labelbottom = False, bottom = False, labeltop = False)
            aux  += 1
            
    plt.savefig(f'False Positive Examples Model: {model_name}',dpi = 200,bbox_inches='tight',pad_inches=0.05)


# ### Filter and filter maps visualization for a simple model 

# In[19]:


def filter_feature_maps_visualization(model,model_name,rows,cols):
    
    for layer in model.layers:
        print(f'CNN1D model layers: {layer}')

    for layer in model.layers:
        if 'conv' not in layer.name:
            continue
        # get filter weights
        filters, biases = layer.get_weights()
        print(layer.name, filters.shape)

    print(len(model.layers[:]))  
    # retrieve weights from the second hidden layer
    filters, biases = model.layers[1].get_weights()

    ix = 1
    for _ in range(rows):
        for _ in range(cols):
            f = filters[ :, :, ix-1] # (3,1,16)
            # specify subplot and turn of axis
            ax = plt.subplot(rows, cols, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.plot(f[:], color='orange')
            plt.title(f'# of filter: {ix}')
            plt.suptitle(f'Feature maps for model: {model_name}')
            plt.grid(alpha=0.5)
            ix += 1
            
    plt.savefig(f'Filters in model {model_name}.png',dpi=200,bbox_inches='tight',pad_inches=0.05)
    
    # Feature map from convolutional layer in model ------------------------------------------------
    model_fm = Model(inputs=model.inputs, outputs=model.layers[1].output)
    model_fm.summary()
    feat_map = model_fm.predict(Xtest)
    print(f'Feature maps shape: {feat_map.shape}')
    
    ix = 1
    for _ in range(rows):
        for _ in range(cols):
            # specify subplot and turn of axis
            ax = plt.subplot(rows, cols, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.plot(feat_map[0, :, ix-1], color='green')
            plt.title(f'Convolved Signal w filter # {ix}',fontsize=8)
            ix += 1
    plt.savefig(f'Feature Map in model {model_name}.png',dpi=200,bbox_inches='tight',pad_inches=0.05)


# In[20]:


# Training and testing the Convolutional Neural Network 

model_name = 'CNN_Reg_1D_3CPDr_FDO_200'

# ---- Training the convolutional Neural Network
#model_cnn,mod_cnn_fit,scores,units,activation,kernel_size,lr = CNN_simple_model(Xtrain,ytrain,Xtest,ytest)
model_cnn,mod_cnn_fit,scores,units,activation,kernel_size,lr,filters = CNN_multiblock_model(Xtrain,ytrain,Xtest,ytest)

# ---- Loss and accuracy functions
plot_loss_acc_func_simple_parameter(mod_cnn_fit,model_name,units,activation,lr,kernel_size,filters)


# In[26]:


# ---- Model evaluation through confusion matrix
p_good,cm = model_test_evaluation(scores,model_cnn,Xtest,ytest,model_name)

# ---- Visualization of misclassified examples from model
misclassified_data(model_name,ytest,p_good,cm)


# ### Performance evaluation of 1D CNN on some examples out of the Training and testing datasets

# In[66]:


# Retreiving data
date = []

date.append('2023_01_01')

data,data_day =retrieve_specific_Xinput_Ytarget(date)


# In[67]:


aux = 0
for i in range(len(data)):
    aux += len(data[i]['Xinput'])
                   
print(aux)
Nsamp_win = len(data[0]['Xinput'][0])
Nlabels   = len(data[0]['Ytarget'][0])

X_input = np.zeros((aux,Nsamp_win))
Y_target = np.zeros((aux))

aux = 0
for i in range(len(data)): # 91 days
    for j in range(len(data[i]['Xinput'])):   # variable lenght, number of 'good' events per day
        X_input[aux,:] = data[i]['Xinput'][j]
        Y_target[aux]  = data[i]['Ytarget'][0][j]
        aux += 1
print(aux)
Xtest_out = X_input
ytest_out = Y_target.reshape(-1).astype(int)

Xtest_out = Xtest_out.reshape(Xtest_out.shape[0],Xtest_out.shape[1],-1)
Xtest_out.shape


# In[68]:


p_test_out = model_cnn.predict(Xtest_out) #.argmax(axis=0)
p_good_out = np.zeros((len(p_test_out[:])),dtype = int)

for i in range(len(p_test_out[:])):
    if p_test_out[i]<=0.5:
        p_good_out[i] = 0
    else:
        p_good_out[i] = 1
print(p_test_out)
#cm = confusion_matrix(ytest_out,p_good_out)
cm = np.zeros((2,2))
print(cm)
cm[0,0] = 0
cm[0,1] = 0
cm[1,0] = 0
cm[1,1] = 5
plot_confusion_matrix(model_name,cm,list(range(2)))


# In[69]:


for i in range(len(p_test_out)):
    print(ytest_out[i] == p_good_out[i])


# In[78]:


# Some misclassified data
fig = plt.figure(figsize=(5,2))
mal_ind = np.where(p_good_out == ytest_out)[0]
i = np.random.choice(mal_ind)
plt.plot(Xtest_out[i],color='magenta')
plt.title('True label: %s , Predicted label: %s, Index: %s' %(ytest_out[i],p_good_out[i],i))


# ### Testing the model on noise data examples from outside the training and testing sets

# In[80]:


# Loading X_input noise
path = 'X_input_mat_regional/'
noise_mat = loadmat(path + 'Xinput_noise_Test_Out.mat')
#noise_mat
x_noise = noise_mat['Xinput_noise']
y_noise = noise_mat['Ytarget_noise'].reshape(-1)
x_noise = x_noise.reshape(x_noise.shape[0],x_noise.shape[1],-1)
x_noise.shape
#y_noise.shape

p_test_out = model_cnn.predict(x_noise)#.argmax(axis=0)
p_good_out = np.zeros((len(p_test_out[:])),dtype = int)

for i in range(len(p_test_out[:])):
    if p_test_out[i]<=0.5:
        p_good_out[i] = 0
    else:
        p_good_out[i] = 1

print(p_test_out)
cm = confusion_matrix(y_noise,p_good_out)
print(cm)
plot_confusion_matrix(model_name,cm,list(range(2)))


# In[92]:


for i in range(5):
    print(y_noise[i] == p_good_out[i])
    
# Some misclassified data
fig = plt.figure(figsize=(5,2))
mal_ind = np.where(p_good_out == y_noise)[0]
i = np.random.choice(mal_ind)
plt.plot(x_noise[i],color='black',alpha=0.5)
plt.title('True label: %s , Predicted label: %s, Index: %s' %(y_noise[i],p_good_out[i],i))


# ## Broader generalization test on January 2023 dataset

# In[93]:


def visualize_TPTS(hours,TPTS_day):
    
    for h in hours:
        print(f'Hour tp: {h} ',TPTS_day[data_day[h] + '_tp'])
        print(f'Hour ts: {h} ',TPTS_day[data_day[h] + '_ts'])


# In[52]:


def retrieve_raw_signal(year,month,day,data_day,index_array):
    #print(data_day,index_array)
    import matlab
    
    signal_raw = {}
    day_py   = matlab.double(int(day))
    
    # For interface this number must be between 1 and 6, that are the indices of an array specific for year 2022
    if month == '08':
        month_py = matlab.double(int(2))
    elif month == '09':
        month_py = matlab.double(int(3)) # corresponds to september
    elif month == '10':
        month_py = matlab.double(int(4))
    elif month == '11':
        month_py = matlab.double(int(5))
    elif month == '12':
        month_py = matlab.double(int(6))
        
    # for 2023:
    if month == '01':
        month_py = matlab.double(int(1))
    elif month == '02':
        month_py = matlab.double(int(2))

    if year == '2022':
        year_py = matlab.double(int(4))
    elif year == '2023':
        year_py = matlab.double(int(5))
    elif year == '2021':
        year_py = matlab.double(int(3))
    elif year == '2020':
        year_py = matlab.double(int(2))

    for h in index_array: # loop for selected hours
        signal_raw[data_day[h]+'_sig0'] = np.array(engine_mat.Interface_from_python(year_py,month_py,day_py,h+1))[0,:,:]    

    return signal_raw


# In[61]:


def NRG_signal_decimated(signal_raw,N,k_samp,index_array,fsamp,fsamp_local):
    from scipy.signal import decimate
    Nr = N - int(10*fsamp) 
    Ndec = int(Nr/k_samp)
    NRG = np.zeros((Nr,24))
    NRG_dec = np.zeros((Ndec,24))
    print(Nr,Ndec)

    for i in (index_array):
        #print(i)
        NRG[:,i] = signal_raw[data_day[i]+'_sig0'][500:359500,3]
        #NRG[:,i] /= np.max(NRG[:,i])
        NRG_dec[:,i] = decimate(NRG[:,i],k_samp)
    
    return NRG_dec, Ndec


# In[55]:


# Initialization of matlab engine for signal retrieving 
#def connect_2_matlab():
import matlab.engine
    
    #global engine_mat
names = matlab.engine.find_matlab()
    
engine_mat = matlab.engine.connect_matlab(names[0])
    
    #return #engine_mat


# In[57]:


os.path.basename("/StorageCitlalli/Paricutin/EnjambreParicutin/Script_sismo_id_h2/res/")
sys.path.append('/StorageCitlalli/Paricutin/EnjambreParicutin/Script_sismo_id_h2/res/')


# In[59]:


# Sampling and resampling parameters
fsamp_orig = 100 #[Hz]
fsamp_local = 20 #[Hz]
k_samp = int(fsamp_orig / fsamp_local)
# Overlapping window parameters
dt = 1/(100/k_samp)   # Fdr decimated signal to 50 Hz == time * 50
wind_time = 37.5   # s
Nsamp_win = int(wind_time/dt)              # Samples number in each window
overlap = 0.5
Nover = int(np.round((Nsamp_win*overlap)))
print(Nsamp_win,Nover)


# In[73]:


""" Main program"""
import time

#connect_2_matlab()

months , day_30, day_31 = month_day_str()
days = []
year = '2023'
 
# testing stuff
monthsss = ['01']
daysssss = ['01']
#daysssss = ['01','02','03','04','05','06']
#daysssss = ['07','08','09','10','11','12']
#daysssss = ['13','14','15','16','17','18','19']
#daysssss = ['20','21','22','23','24','25']
#daysssss = ['26','27','28','29','30']
#daysssss = ['06','07','08','09','10','11','12'] 
count_total = 0

index_array = [i for i in range(0,1)]
#print(index_array)
#start_time = time.time()   # Timing process

for month in monthsss:
    if  month == '01' or month == '03' or month == '05' or month == '07'     or month == '08' or month == '10' or month == '12':
        days = copy.copy(day_31)
    elif month == '04' or month == '06' or month == '09' or month == '11':
        days = copy.copy(day_30)
  
    for day in daysssss:#days[:]:
        data_day , TPTS_day, hours = matfiles_retrieve_24(year,month,day)
        
        visualize_TPTS(hours,TPTS_day)
        data_month = data_day[0][0:10]

        """Calling SeisPick program in order to retrieve specific hours with false detections
        Testing calls from here to Matlab seismicpick program interface through function
        Retrieving Raw Signal"""
        
        #print(index_array)
        print(f'\n---------Into Signal Retrieval... Loading Signals {data_month}...-----------\n')
        signal_raw = retrieve_raw_signal(year,month,day,data_day,index_array)

        """ Structures for component retrieve: Signal energy + normalization + decimate """
        if index_array[0] <= 9: 
            N = len(signal_raw[data_month + '_0' + str(index_array[0]) + '_sig0'])
        else:
            N = len(signal_raw[data_month + '_' + str(index_array[0]) + '_sig0'])
            
        NRG_dec, Ndec = NRG_signal_decimated(signal_raw,N,k_samp,index_array,fsamp_orig,fsamp_local)
        
        print(NRG_dec[:,0],Ndec,NRG_dec.shape)
        plt.figure()
        plt.plot(NRG_dec[:,0],color='red')
        
        nwin_sig = int(np.floor(Ndec/Nsamp_win))-1
        wind_sig = np.zeros((nwin_sig,Nsamp_win,1))
        
        for h in index_array:
            for i in range(nwin_sig):
                #wind_sig[i,:,0] = NRG_dec[(5*fsamp_local)+i*Nsamp_win:Nsamp_win*(i+1)+(5*fsamp_local),h]
                wind_sig[i,:,0] = NRG_dec[i*Nsamp_win:Nsamp_win*(i+1),h]/np.max(NRG_dec[i*Nsamp_win:Nsamp_win*(i+1),h])
#                 plt.figure()
#                 plt.plot(wind_sig[i,:,0])
#                 plt.title(f'Window signal hour:{h}, nwindow: {i}')
        
            wind_sig.reshape(wind_sig.shape[0],wind_sig.shape[1],-1)
            print(wind_sig.shape)

            proba_class = model_cnn.predict(wind_sig)
            prediction_class = np.zeros((len(proba_class[:])),dtype = int)

            for k in range(len(proba_class[:])):
                if proba_class[k]<=0.5:
                    prediction_class[k] = 0
                else:
                    prediction_class[k] = 1

                print(k,proba_class[k],prediction_class[k])
                

#         """ Time vector, efective number of signal hours and count of events """
#         time , Nhours = time_windowing(Ndec,Nsamp_win,Nover,overlap,index_array)
#         count_event = count_local_events(index_array,TPTS_day)
#         print(f'Count Events for {data_month}: {count_event} \n')
        
# #         Counting total events from september to november!
#         count_total += count_event
#         print(f'-------------Total events to {data_month}: {count_total}------------------')

#         """ X input windows visualization """
#         opc = True
#         if opc:
#             Xinput_figure_2(Nsamp_win,count_event,data_month,TPTS_day,NRG_dec,time,fsamp_local,Nover,index_array)
        
#         """ X input and Y target saving into .mat structures """
#         opc = True
#         if opc == True:
#             X_input_Y_target(Nsamp_win,Nhours,count_event,index_array,TPTS_day,data_day,NRG_dec,fsamp_local,Nover,data_month)

#engine_mat.quit()
#print("---X input, y target Process execution time: %s seconds ---" % (time.time() - start_time))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[72]:


# ---- Visualization of filters structure and feature maps
if filters == 16:
    rows,cols = 4,4
elif filters == 32:
    rows,cols = 8,4 
elif filters == 64:
    rows,cols = 8,8
    
filter_feature_maps_visualization(model_cnn,model_name,rows,cols)


# In[206]:


fig1 = plt.figure(figsize=(10,5))
plt.plot(Xtest[0,:])
plt.title(f'Original Xtest signal, Example {0}')


# In[ ]:



# # plot feature map of first conv layer for given image
# from keras.applications.vgg16 import VGG16
# from keras.applications.vgg16 import preprocess_input
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from keras.models import Model
# from matplotlib import pyplot
# from numpy import expand_dims
# # load the model
# model = VGG16()
# # redefine model to output right after the first hidden layer
# model = Model(inputs=model.inputs, outputs=model.layers[1].output)
# model.summary()
# # load the image with the required shape
# img = load_img('bird.jpg', target_size=(224, 224))
# # convert the image to an array
# img = img_to_array(img)
# # expand dimensions so that it represents a single 'sample'
# img = expand_dims(img, axis=0)
# # prepare the image (e.g. scale pixel values for the vgg)
# img = preprocess_input(img)
# # get feature map for first hidden layer
# feature_maps = model.predict(img)
# # plot all 64 maps in an 8x8 squares
# square = 8
# ix = 1
# for _ in range(square):
#     for _ in range(square):
#          # specify subplot and turn of axis
#          ax = pyplot.subplot(square, square, ix)
#          ax.set_xticks([])
#          ax.set_yticks([])
#          # plot filter channel in grayscale
#          pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
#          ix += 1
# # show the figure
# pyplot.show()


# ## FIRST ARCHITECTURE HYPERPARAMETER TUNING
# * Set the best CNN hyperparameter tuning
# * Convolutional - Pooling - Flatten - Dense - Output

# In[142]:


# 1D Convolutional Neural Network for seismic time series classification
def CNN_model_evaluation(Xtrain,ytrain,Xtest,ytest,param_num,parameter):
    import time
    input_shape = tf.TensorShape(Xtrain[0].shape)
    units = 64
    nclass = 1
    pool_size = 2
    activation = 'sigmoid'
    
    if parameter == 'filters':
        filters = param_num
        kernel_size = 3
        lr = 0.0001
    elif parameter == 'kernel_size':
        kernel_size = param_num
        filters = 32
        lr = 0.0001
    elif parameter == 'learning_rate':
        filters = 32
        kernel_size = 3
        lr = param_num

    i = Input(shape=input_shape)
    x = Conv1D(filters=filters,kernel_size=kernel_size,activation=activation)(i) # Sigmoid for Classification
    x = MaxPool1D(pool_size = pool_size)(x)
    x = Flatten()(x)
    x = Dense(units,activation = activation)(x)
    x = Dense(nclass,activation='sigmoid')(x)

    model = Model(i,x)
    #model.summary()

    # Training the convolutional neural network
    start_time = time.time()

    # Stablish the model compile parameters
    model.compile(loss='binary_crossentropy',optimizer=optimizers.Adam
                  (learning_rate=lr),metrics = 'accuracy')
    verbose = 0
    mod_cnn = model.fit(Xtrain,ytrain,validation_data=(Xtest,ytest),epochs=100,verbose=verbose)
    score   = model.evaluate(Xtest,ytest,verbose=verbose)

    print("---CNN1D training and testing time execution: %s seconds ---" % (time.time() - start_time))
    
    return mod_cnn,score


# In[143]:


def stat_results(model_name,accur,loss,params,parameter):
    
    # summarize mean and standard deviation
    for i in range(len(params)):
        mean_acc, std_acc = np.mean(accur[i]), np.std(accur[i])
        mean_loss,std_loss = np.mean(loss[i]), np.std(loss[i])
        #print('Param=%d: %.3f%% (+/-%.3f)' % (params[i], mean_acc, std_acc))
        print(f'Parameter {parameter}: {params[i]} , Mean: {(mean_acc*100):.3f}%, Std: (+/-{std_acc*100:.3f})')
        
    # boxplot of scores
    plt.figure(figsize=(10,5))
    plt.boxplot(accur, labels=params)
    plt.title(f'Model {model_name}: Boxplot for Parameter {parameter} Variation, Accuracy')
    plt.xlabel(f'{parameter}')
    plt.ylabel('Accuracy')
    plt.savefig(f'cnn1D_{parameter}_variable.png')


# In[144]:


def plot_loss_acc_func_variable_parameter(model_reps,reps,parameters):
    
    mpl.rcParams['figure.figsize'] = [15,8]
    fig,ax = plt.subplots(2,2)
    for p in range(len(parameters)):
        for r in range(reps):
            plt.suptitle(f'{model_name} Loss and accuracy functions, Units: {units}, Filters: {filters}, {activation},                     Out: {activation}, Kernel Size {parameters}',fontsize=12,weight='bold')
            ax[0,0].title.set_text('Loss Training')
            ax[0,0].plot(model_reps[p][r].history['loss'],linewidth=2.5,marker = '.',label='train_loss')
            #ax[0,0].legend()
            ax[0,0].grid(alpha=0.5)

            ax[1,0].title.set_text('Accuracy Training')
            ax[1,0].plot(model_reps[p][r].history['accuracy'],linewidth=2.5,marker='.',label='train_acc')#color='green'
            ax[1,0].set_xlabel('Epochs',fontsize=12,weight='bold')
            #ax[1,0].legend()
            ax[1,0].grid(alpha=0.5)

            ax[0,1].title.set_text('Loss Testing')
            ax[0,1].plot(model_reps[p][r].history['val_loss'],label='test_loss',linewidth=2.5,linestyle='-')
            #ax[0,1].legend()
            ax[0,1].grid(alpha=0.5)

            ax[1,1].title.set_text('Accuracy Testing')
            ax[1,1].plot(model_reps[p][r].history['val_accuracy'],label='test_acc',linewidth=2.5,linestyle='-')
            ax[1,1].set_xlabel('Epochs',fontsize=12,weight='bold')
            #ax[1,1].legend()
            ax[1,1].grid(alpha=0.5)

    # fig.savefig(f'{model_name}_Loss_Accuracy_Adam_Units{str(units)}_Filters{str(filters)}_{activation}.png'\
    #             ,bbox_inches='tight',pad_inches=0,dpi=100)


# In[211]:


model_name = 'CNN_Reg_1D_CPFDO_Hyp_Tune'
#parameter = input('Enter the Parameter name as follows: \n kernel_size \n filters \n learning_rate  ' )
#parameter = 'kernel_size'
#parameter = 'filters'
parameter = 'learning_rate'

if parameter == 'kernel_size':
    n_params = [2,3,5,7,9,11]
elif parameter == 'filters':
    n_params = [8,16,32,64,128,256]
elif parameter == 'learning_rate':
    n_params = [0.1,0.01,0.001,0.0001,0.00001]
reps = 5
accur_reps = []
loss_reps = []
model_reps = []

for k in n_params:
    losses = []
    accurs = []
    models = []
    for r in range(reps):
        mod_cnn,score = CNN_model_evaluation(Xtrain,ytrain,Xtest,ytest,k,parameter)
        
        models.append(mod_cnn)
        losses.append(score[0])
        accurs.append(score[1])
        print(f'Parameter: {k}, Scores: Loss {score[0]:.3f}, Accuracy {score[1]:.3f}, Rep: {r+1}')
        
    model_reps.append(models)
    accur_reps.append(accurs)
    loss_reps.append(losses)
    
stat_results(model_name,accur_reps,loss_reps,n_params,parameter)


# In[213]:


#plot_loss_acc_func_variable_parameter(model_reps,reps,n_params)


# In[214]:


#parameter = 'kernel_size'
#stat_results(model_name,accur_reps,loss_reps,n_params,parameter)


# In[ ]:


parameter = 'filters'
stat_results(model_name,accur_reps,loss_reps,n_params,parameter)


# In[19]:


units = 32
filters = 32
activation = 'sigmoid'

#plot_loss_acc_func_variable_parameter(model_reps,reps,n_kernels)


# In[20]:


# # Train test splitting using SKlearn
# from sklearn.model_selection import train_test_split
# test_s = 0.3
# Xtrain, Xtest, ytrain, ytest = train_test_split(X_input, Y_target, test_size=test_s, random_state=42)

# Ninput = X_input.shape[0]
# Nwin = X_input.shape[1]

# perc_train = 1.0 - test_s
# perc_test  = test_s

# Ntrain = int(Ninput * perc_train)
# Ntest  = int(Ninput * perc_test )
# Xtrain = Xtrain.reshape((Ntrain, Nwin, -1))
#Xtest = Xtest.reshape((Ntest, Nwin, -1))

# Class transform to matrix form using one-hot encoding
# from tensorflow.keras.utils import to_categorical
# ytrain = to_categorical(ytrain)
# ytest  = to_categorical(ytest)


# In[ ]:


# plt.figure(figsize=(10,10))
# for i in range(3):
#     plt.plot(model_reps[1][i].history['loss'])
#     plt.plot(model_reps[0][i].history['val_loss'])
# plt.figure()
# for i in range(3):
#     plt.plot(model_reps[0][i].history['accuracy'])
# plt.figure()
# for i in range(3):
#     plt.plot(model_reps[0][i].history['val_loss'])
# plt.figure()
# for i in range(3):
#     plt.plot(model_reps[0][i].history['val_accuracy'])


# ### Hyperparameter Optimal CNN 1D

# In[215]:


#import visualkeras
#from PIL import ImageFont
model_name = 'CNN_Regional_1D_CPFDO_OPT'
input_shape = tf.TensorShape(Xtrain[0].shape)
units = 64
nclass = 1
pool_size = 2
activation = 'sigmoid'

# Optimal hyperparameters
filters = 32
kernel_size = 11
lr = 0.0001

i = Input(shape=input_shape)
x = Conv1D(filters=filters,kernel_size=kernel_size,activation=activation)(i) # Sigmoid for Classification
x = MaxPool1D(pool_size = pool_size)(x)
x = Flatten()(x)
x = Dense(units,activation = activation)(x)
x = Dense(nclass,activation='sigmoid')(x)

model = Model(i,x)
model.summary()
#visualkeras.layered_view(model, legend=True)


# In[216]:


# Training the convolutional neural network
start_time = time.time()

# Stablish the model compile parameters
model.compile(loss='binary_crossentropy',optimizer=optimizers.Adam
                  (learning_rate=lr),metrics = 'accuracy')
verbose = 2
mod_cnn = model.fit(Xtrain,ytrain,validation_data=(Xtest,ytest),epochs=100,verbose=verbose)

print("---CNN1D training and testing time execution: %s seconds ---" % (time.time() - start_time))


# In[217]:


score   = model.evaluate(Xtest,ytest,verbose=verbose)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[257]:


plot_loss_acc_func_simple_parameter(mod_cnn,model_name,units,activation,lr,kernel_size)


# In[220]:


# Test evaluation through Confusion matrix
p_test = model.predict(Xtest)#.argmax(axis=0)
p_test.shape

print(f'Statistical Summary of Model predictions: \n Mean = {np.mean(p_test)} \n Min = {p_test.min()} \n Max = {p_test.max()}')

plt.figure(figsize=(3,3))
plt.hist(p_test)
plt.grid(alpha=0.5)
plt.title('Probability histogram for Test data')
# penguins = sns.load_dataset("penguins")
# sns.histplot(data=penguins, x="flipper_length_mm")

p_good = np.zeros((len(p_test[:])),dtype = int)

for i in range(len(p_test[:])):
    if p_test[i]<=0.5:
        p_good[i] = 0
    else:
        p_good[i] = 1
p_good
cm = confusion_matrix(ytest,p_good)
plot_confusion_matrix(model_name,cm,list(range(2)))


# In[223]:


# Some misclassified data
#fig = plt.figure(figsize=(15,6))
#model_name = 'CNN_Reg_1D_CPFDO'
mal_ind = np.where(p_good != ytest)[0]
mal_ind = np.unique(mal_ind)

mpl.rcParams['figure.figsize'] = [11,11]
fig,ax = plt.subplots(7,4,layout = 'constrained')

aux = 0
for i in range(7):
    for j in range(4):
        if ytest[mal_ind[aux]] == 0:
            col = 'green'
        else:
            col = 'red'
            
        ax[i,j].plot(Xtest[mal_ind[aux]],color=col)
        ax[i,j].set_title('True label: %s , Predicted label: %s, Index: %s'                           %(ytest[mal_ind[aux]],p_good[mal_ind[aux]],mal_ind[aux]),fontsize=8)

        ax[i,j].tick_params(left = False, right = False , labelleft = False ,
                     labelbottom = False, bottom = False, labeltop = False)
        aux  += 1
plt.savefig(f'False Positive Examples Model: {model_name}',dpi = 200,bbox_inches='tight',pad_inches=0.05)


# In[245]:


for layer in model.layers:
    print(f'CNN1D Opt model layers: {layer}')

for layer in model.layers:
    if 'conv' not in layer.name:
        continue
    # get filter weights
    filters, biases = layer.get_weights()
    print(layer.name, filters.shape)
    
print(len(model.layers[:]))  


# ### Checking the Convolution Layer

# In[244]:


# retrieve weights from the second hidden layer
filters, biases = model.layers[1].get_weights()

#print(f'filters: {filters}, Num of Filters = {len(filters[0][0])}')
print(f'Num of Filters = {len(filters[0][0])}')

# plot first few filters
n_filters, ix = 32, 1
rows,cols = 8,4

for _ in range(rows):
    for _ in range(cols):
        f = filters[ :, :, ix-1] # (3,1,16)
        # specify subplot and turn of axis
        ax = plt.subplot(rows, cols, ix)
        ax.set_xticks([])
        #ax.set_yticks([])
        # plot filter channel in grayscale
        plt.plot(f[:], color='orange',marker='*')
        plt.title(f'Filter: {ix}',fontsize=10)
        plt.grid(alpha=0.5)
        ix += 1


# In[238]:


model_fm = Model(inputs=model.inputs, outputs=model.layers[1].output)
model_fm.summary()
feat_map_opt = model_fm.predict(Xtest)
print(f'Feature maps shape: {feat_map_opt.shape}')


# In[255]:


fig1 = plt.figure(figsize=(10,5))
plt.plot(Xtest[508,:])
plt.title(f'Original Xtest signal, Example {508}, Class: {ytest[508]}')


# In[253]:



n_filters, ix = 32, 1
rows,cols = 8,4
for _ in range(rows):
    for _ in range(cols):
        # specify subplot and turn of axis
        ax = plt.subplot(rows, cols, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        plt.plot(feat_map_opt[508, :, ix-1], color='green')
        plt.title(f'Convolved Signal w filter # {ix}',fontsize=8)
        ix += 1


# ### Checking the Pooling Layer

# In[256]:


# retrieve weights from the third hidden layer
model.layers[2]


# In[248]:


model_pool = Model(inputs=model.inputs, outputs=model.layers[2].output)
model_pool.summary()
pool_opt = model_pool.predict(Xtest)
print(f'Pool shape: {pool_opt.shape}')


# In[254]:


n_filters, ix = 32, 1
rows,cols = 8,4
for _ in range(rows):
    for _ in range(cols):
        # specify subplot and turn of axis
        ax = plt.subplot(rows, cols, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        plt.plot(pool_opt[508, :, ix-1], color='purple')
        plt.title(f'Pooled Signal of filter # {ix}',fontsize=8)
        ix += 1

