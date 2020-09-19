#!/usr/bin/env python
# coding: utf-8

# ## Connect GoogleColab

# ## 一、載入相關套件

# In[1]:


# 資料處理套件
import cv2
import csv
import os
import random
import time
import numpy as np
import pandas as pd
import matplotlib.image as mpimg # mpimg 用於讀取圖片
import matplotlib.pyplot as plt # plt 用於顯示圖片
import seaborn as sns


# In[2]:


# 設定顯示中文字體
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 


# In[3]:


# Keras深度學習模組套件
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras import utils as np_utils
from keras import backend as K
from keras import optimizers


# In[4]:


# tensorflow深度學習模組套件
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf


# In[5]:


# 打印相關版本
print(pd.__version__)
print(tf.__version__)


# In[6]:


# 查看通道位置
print(K.image_data_format())


# ## 二、讀取資料

# In[7]:


# 讀取資料集標籤檔
Sample_label = pd.read_csv("train.csv",encoding="utf8")


# In[8]:


# 顯示資料集標籤檔
Sample_label.head()


# In[9]:


# 串接圖片檔的路徑
Sample_pics_path = os.path.join("Train")


# In[10]:


# 讀取路徑中的圖片
train_mango_fnames = os.listdir(Sample_pics_path)


# In[11]:


# 檢視是否有讀取到圖片
print(train_mango_fnames[0])
print(train_mango_fnames[1])
print(train_mango_fnames[2])


# In[12]:


# 用於瀏覽標籤檔的概況
label_Survey = pd.read_csv("train_survey.csv",encoding="utf8")


# In[13]:


label_Survey.head()


# In[14]:


sns.countplot(label_Survey['label'], hue = label_Survey["label"])


# In[15]:


sector = label_Survey.groupby('label')
sector.size()


# ## 三、顯示芒果圖片

# In[16]:


# 讀取圖檔
img = mpimg.imread("Train/00002.jpg")
# 查看資料型態
type(img)


# In[17]:


# 顯示圖片的比例
img.shape


# In[18]:


# 把圖片的比例壓縮至800x800 
res = cv2.resize(img,(800,800),interpolation=cv2.INTER_LINEAR)


# In[19]:


# 顯示壓縮後圖片的比例
res.shape


# In[20]:


# 顯示原圖的芒果照片
plt.imshow(img)
plt.axis('off')
plt.show()


# In[21]:


# 顯示壓縮過原圖的芒果照片
plt.imshow(res)
plt.axis('off')
plt.show()


# ## 四、製作標籤&資料集

# In[22]:


csvfile = open('train.csv',encoding="utf-8")
reader = csv.reader(csvfile)


# In[23]:


# 讀取csv標籤
labels = []
for line in reader:
    tmp = [line[0],line[1]]
    # print tmp
    labels.append(tmp)

csvfile.close() 


# In[24]:


picnum = len(labels)
print("芒果圖片數量: ",picnum)


# In[25]:


labels[8]


# In[26]:


X = []
y = []


# In[27]:


# 轉換圖片的標籤
for i in range(len(labels)):
    labels[i][1] = labels[i][1].replace("A","0")
    labels[i][1] = labels[i][1].replace("B","1")
    labels[i][1] = labels[i][1].replace("C","2")


# In[28]:


# 隨機讀取圖片
a = 0
items= []


# In[29]:


import random
for a in range(0,5600):
    items.append(a)


# In[30]:


# 製作訓練用資料集及標籤
for i in random.sample(items,5600):
    img = cv2.imread("Train/" + labels[i][0] )
    res = cv2.resize(img,(108,108),interpolation=cv2.INTER_LINEAR)
    res = img_to_array(res)
    X.append(res)    
    y.append(labels[i][1])


# In[31]:


y_label_org = y


# In[32]:


img[0]


# In[33]:


print(len(X))
print(len(y))


# In[34]:


# 轉換至array的格式
X = np.array(X)
y = np.array(y)


# In[35]:


# 轉換至float的格式
for i in range(len(X)):
    X[i] = X[i].astype('float32')


# In[36]:


# 打映圖片訓練集的概況
# print(X[0])
print(type(X))
print(X.shape)

print(X[0].shape)
print(type(X[0]))


# In[37]:


# 將標籤轉換至float格式
y = tf.strings.to_number(y, out_type=tf.float32)


# In[38]:


# 打映圖片標籤的概況
print(y[0])
print(type(y[0]))


# In[39]:


# 標籤進行one-hotencoding
y = tf.keras.utils.to_categorical(y, num_classes = 3)


# In[40]:


y[0]


# ## 五、製作訓練資料集

# In[41]:


# 分配訓練集及測試集比例
x_train = X[:5000]
y_train = y[:5000]
x_test = X[5000:]
y_test = y[5000:]


# In[42]:


y_test


# In[43]:


y_train_label = [0.,0.,0.]

for i in range(0,len(y_train)):
    y_train_label = y_train[i] + y_train_label


# In[44]:


y_test_label = [0.,0.,0.]

for i in range(0,len(y_test)):
    y_test_label = y_train[i] + y_test_label


# In[45]:


y_train_label


# In[46]:


y_test_label


# In[47]:


print(type(x_train))
print(len(x_train))
print(x_train.shape)
print(type(x_train[0]))


# ## 六、建立與訓練深度學習Model

# In[48]:


# 建立深度學習CNN Model

model = tf.keras.Sequential()

model.add(layers.Conv2D(16,(3,3),
                 strides=(1,1),
                 input_shape=(108, 108, 3),
                 padding='same',
                 activation='relu',
                 ))

model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D(pool_size=(2,2),strides=None))

#model.add(layers.Flatten())

model.add(layers.Dropout(0.25))
###
model.add(layers.Conv2D(32,(3,3),
                 strides=(1,1),
                 padding='same',
                 activation='relu',
                 ))

model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D(pool_size=(2,2),strides=None))

#model.add(layers.Flatten())

model.add(layers.Dropout(0.25))
###
model.add(layers.Conv2D(64,(3,3),
                 strides=(1,1),
                 padding='same',
                 activation='relu',
                 ))

model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D(pool_size=(2,2),strides=None))

#model.add(layers.Flatten())

model.add(layers.Dropout(0.25))
###
model.add(layers.Conv2D(128,(3,3),
                 strides=(1,1),
                 padding='same',
                 activation='relu',
                 ))

model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D(pool_size=(2,2),strides=None))

#model.add(layers.Flatten())

model.add(layers.Dropout(0.25))
###
'''
model.add(layers.Conv2D(256,(3,3),
                 strides=(1,1),
                 padding='same',
                 activation='relu',
                 ))

model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D(pool_size=(2,2),strides=None))

#model.add(layers.Flatten())

model.add(layers.Dropout(0.2))
'''
###

model.add(layers.Flatten())

model.add(layers.Dropout(0.25))

model.add(layers.Dense(64,activation='relu'))

model.add(layers.Dense(128,activation='relu'))


model.add(layers.Dropout(0.25))

model.add(layers.Dense(3,activation='softmax'))

model.summary()


# history = model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#               metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# In[49]:


adam = optimizers.adam(lr=0.05, epsilon = None) # decay=0.01
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['acc'])


# In[50]:


# zca_whitening 對輸入數據施加ZCA白化
# rotation_range 數據提升時圖片隨機轉動的角度
# width_shift_range 圖片寬度的某個比例，數據提升時圖片水平偏移的幅度
# shear_range 剪切強度（逆時針方向的剪切變換角度）
# zoom_range 隨機縮放的幅度
# horizontal_flip 進行隨機水平翻轉
# fill_mode ‘constant’，‘nearest’，‘reflect’或‘wrap’之一，當進行變換時超出邊界的點將根據本參數給定的方法進行處理

datagen = ImageDataGenerator(
    zca_whitening=False,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')


# In[51]:


# 導入圖像增強參數
datagen.fit(x_train)
x_train = x_train/255
x_test = x_test/255
print('rescale！done!')


# In[52]:


# 設定超參數HyperParameters 
batch_size =  64 #or64
epochs = 70


# In[53]:


# 檔名設定
file_name = str(epochs)+'_'+str(batch_size)


# In[54]:


# 加入EarlyStopping以及Tensorboard等回調函數
CB = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10) #在10個epoch後loos沒下降就停止
TB = keras.callbacks.TensorBoard(log_dir='./log'+"_"+file_name, histogram_freq=1)


# In[56]:


history = model.fit(
    x = x_train , y = y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_split = 0.2,
    #callbacks = [CB]
)


# ## 柒、繪製Model學習成效

# In[77]:


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()


# In[78]:


plot_learning_curves(history)


# In[ ]:


# 绘制训练 & 验证的准确率值
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[ ]:


# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# ## 捌、推測圖片

# In[61]:


test_mango_dir = os.path.join("test_image")
test_mango_fnames = os.listdir(test_mango_dir)


# In[62]:


test_mango_fnames[0]


# In[63]:


img_files = [os.path.join(test_mango_dir,f) for f in test_mango_fnames]
img_path = random.choice(img_files)

# 讀入待測試圖像並秀出
img = load_img(img_path, target_size=(800, 800))  # this is a PIL image
plt.title(img_path)
plt.grid(False)
plt.imshow(img)


# In[64]:


labels = ['A','B',"C"]


# ## 玖、測試集預測準確度

# In[66]:


# 測試集標籤預測
y_pred = model.predict(x_test)


# In[67]:


# 整體準確度
count = 0
for i in range(len(y_pred)):
    if(np.argmax(y_pred[i]) == np.argmax(y_test[i])): #argmax函数找到最大值的索引，即为其类别
        count += 1
score = count/len(y_pred)
print('Accuracy:%.2f%s' % (score*100,'%'))


# In[68]:


# 模型預測後的標籤
predict_label = np.argmax(y_pred,axis=1)


# In[69]:


# 模型原標籤
true_label = y_label_org[5000:]
true_label = list(map(int, true_label))
true_label = np.array(true_label)


# In[70]:


# 模型預測後的標籤
predictions = model.predict_classes(x_test)


# In[71]:


print(pd.crosstab(true_label,predict_label,rownames=['實際值'],colnames=['預測值']))


# In[72]:


# 儲存模型相關參數
# model.save('h5/'+file_name+'.h5')


# In[74]:


from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
print("Recall : ")
print(recall_score(true_label, predictions, average=None))
print("WAR : ")
print(recall_score(true_label, predictions, average='weighted'))


# In[ ]:




