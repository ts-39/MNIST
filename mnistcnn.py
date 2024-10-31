import keras
from keras.datasets import mnist
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten

from tensorflow.keras.optimizers import Adam

import time


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# print(x_train.shape)
# (48000, 28, 28)
# print(x_test_origin.shape)
# (10000, 28, 28)
# print(x_val.shape)
# (12000, 28, 28)
# print(x_train[0].shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_val = x_val.astype('float32')

x_train /= 255
x_test /= 255
x_val /= 255

y_train  = keras.utils.to_categorical(y_train, 10)
y_test   = keras.utils.to_categorical(y_test, 10)
y_val   = keras.utils.to_categorical(y_val, 10)

model = Sequential()
model.add(Conv2D(16, (3, 3), padding='same', input_shape=(28, 28, 1), activation='relu'))
# padding = 'same'はフィルタした時に余る周りの枠を０で埋める→常にインプットとアウトプットが同じ値になる
# input_shape=(28, 28, 1)は、28*28ピクセルのグレースケール。
# input_shape=(150,150,3)は、150*150ピクセルのRGB画像
model.add(MaxPooling2D(pool_size=(2, 2)))
#プーリング層の目的は計算のふか、メモリ使用量、パラメータ数の削減。過学習の緩和の効果。のために入力画像をサブサンプリング
#フィルタから最大値とか平均をとって画像を小さくする。→non-trainable
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
#フィルタ数はどんどんん増えていく。低水準特徴量が積み重なっていくから
# プーリング層を通過するたびにフィルタ数を2倍にするのが主流
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
# 0.5の確率でニューロンの値を0にする。CNNでは40-50%,RNN=20-30%,
# 0.5で訓練した時、テストでは、それがされないため、訓練とは2倍の入力になってしまうため、出力に0.5をかける、もしくは訓練中に2倍する。
# model.add(Dropout(0.5))のコードでは、0.5の確率で0にし、k訓練中に値を(1-0.5)で割っている。
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

start_time = time.time()

# history = model.ﬁt(x_train, y_train, batch_size=2048, epochs=100, verbose=1, validation_data=(x_val, y_val))

# print(model.summary())

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Computation time:{0:.3f} sec'.format(time.time() - start_time))

predict = model.predict(x_test, batch_size=256)


# print()