#--------------------------------------#
#  Binary classification(二値分類)　　　　#
#　-------------------------------------#

from keras.datasets import imdb


#-------------------------------------#
#  データの読み込み　　　              #
#-------------------------------------#
(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=1000)

#-------------------------------------#
#  データのベクトル化　　              #
#-------------------------------------#
import numpy as np

def vectorize_sequences(sequences,dimension=1000):
    results=np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence]=1.
    return results

#訓練・テストデータのベクトル化
x_train=vectorize_sequences(train_data)
x_test=vectorize_sequences(test_data)

#ラベルもベクトル化
y_train=np.asarray(train_labels).astype('float32')
y_test=np.asarray(test_labels).astype('float32')


#-------------------------------------#
#  NNの構築　　　　　　　              #
#-------------------------------------#
from keras import models
from keras import layers

model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(1000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))


#-------------------------------------#
#  　損失関数とオプティマイザ          #
#-------------------------------------#
from keras import optimizers
model.compile(optimizer='rmsprop',
loss='binary_crossentropy',
metrics=['accuracy'])

#-------------------------------------#
#  検証データの作成　　　　　          #
#-------------------------------------#
x_val=x_train[:1000]
partial_x_train=x_train[1000:]

y_val=y_train[:1000]
partial_y_train=y_train[1000:]

#-------------------------------------#
#  モデルの訓練　　　　　　　          #
#-------------------------------------#
model.compile(optimizer='rmsprop',
loss='binary_crossentropy',
metrics=['acc'])

history=model.fit(partial_x_train,
partial_y_train,
epochs=4,
batch_size=512,
validation_data=(x_val,y_val))

#--------------------------------------------#
#  訓練データと検証データでの損失値をプロット　#
#--------------------------------------------#

import matplotlib.pyplot as plt

history_dict=history.history
loss_values=history_dict['loss']
val_loss_values=history_dict['val_loss']

epochs=range(1,len(loss_values)+1)

plt.plot(epochs,loss_values,'bo',label='Training loss')
plt.plot(epochs,val_loss_values,'b',label='Validation loss')
plt.title('Training and Varidation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#--------------------------------------------#
#  訓練データと検証データでの正解率をプロット　#
#--------------------------------------------#

plt.clf()
acc=history_dict['acc']
val_acc=history_dict['val_acc']

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#--------------------------------------------#
#  学習済みネットワークを利用する　　          #
#--------------------------------------------#

print(model.predict(x_test))

