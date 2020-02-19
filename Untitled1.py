#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas_profiling as pp

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score, precision_score, recall_score


# In[4]:


df = pd.read_csv(r'C:\Users\gsvks\OneDrive\google\musk_csv.csv')
df.head()


# In[5]:


#Checking Null values
df.isna().sum()


# In[6]:


# Creating our correlation matrix for finding better relationship between data points
corr_matrix = df.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.92
to_drop = [column for column in upper.columns if any(upper[column] > 0.92)]


# In[7]:


df1 = df.drop(columns = to_drop)


# In[8]:


df1.shape


# In[9]:


train,test = train_test_split(df1, random_state=30, test_size = 0.2)
Xtrain = train.iloc[:,3:-1]
Ytrain = train.iloc[:,-1:]
Xtest = test.iloc[:,3:-1]
Ytest = test.iloc[:,-1:]
Xtrain.shape


# In[10]:


pip install keras


# In[11]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf


# In[12]:


a=[1]*Xtrain.shape[0]
Xtrain["demo"]=a
Xtrain.shape


# In[13]:


b=[1]*Xtest.shape[0]
Xtest["demo"]=b
Xtest.shape


# In[14]:


x_train=Xtrain.values.reshape(Xtrain.shape[0],19,6,1)
x_test=Xtest.values.reshape(Xtest.shape[0],19,6,1)


# In[15]:


x_train.shape


# In[16]:


x_test.shape


# In[17]:


model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(19,6,1)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))


# In[18]:


model.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])


# In[19]:


history = model.fit(x_train,Ytrain,batch_size=128,epochs= 20,validation_data=(x_test,Ytest)) 
score=model.evaluate(x_test,Ytest,verbose=0)
print(score)


# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt


# In[22]:


# summarize history for accuracy
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()

# summarize history for loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper right")
plt.show()


# In[23]:


print("f1_score:",f1_score(Ytest,model.predict_classes(x_test),))
print("recall:",recall_score(Ytest,model.predict_classes(x_test),))
print("Validation Loss:",score[0])
print("Validation Accuracy:",score[1])


# In[27]:


model.save(r"C:\Users\gsvks\OneDrive\Desktop\New Rich Text Document")


# In[ ]:




