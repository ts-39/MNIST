import datetime
import keras
from keras.datasets import mnist
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer
import numpy as np
import math

# from exec_opts import ExecOpts
# from mnist_cnn_keras import mnist_cnn

(x_train, y_train), (x_test_origin, y_test) = keras.datasets.mnist.load_data()

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# print(x_train.shape)
# print(x_test.shape)
# print(x_val.shape)
# print(y_val.shape)

x_train = x_train.reshape(48000, 784)
x_test = x_test_origin.reshape(10000, 784)
x_val = x_val.reshape(12000, 784)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_val = x_val.astype('float32')

x_train /= 255
x_test /= 255
x_val /= 255

y_train  = keras.utils.to_categorical(y_train, 10)
y_test   = keras.utils.to_categorical(y_test, 10)
y_val   = keras.utils.to_categorical(y_val, 10)
# print(x_train.shape)

model = Sequential()
model.add(InputLayer(input_shape=(784,)))
# model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


epochs = 100
batch_size = 2048
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val, y_val))
# history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)


score = model.evaluate(x_test, y_test, verbose=1)
print(score)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# batch_size = 128
# Test loss: 2.3576579093933105
# Test accuracy: 0.12630000710487366

# batch_size = 2048
# Test loss: 0.2842666804790497
# Test accuracy: 0.9205999970436096

predict = model.predict(x_test, batch_size=256)
pre_ans = predict.argmax(axis=1)

y_test_change = np.argmax(y_test, axis=1) #onehotを普通のにもどす
a = y_test_change == pre_ans


false_list = []
for i in range(len(a)):
    item = a[i]
    if item == False:
        false_list.append(i)

false_pic_list = []
for i in false_list:
    false_pic_list.append(x_test_origin[i])

print(len(false_pic_list))

# print(type(false_pic_list))
false_pic = np.array(false_pic_list)
# print(type(false_pic))

import keras
from keras.datasets import mnist
import numpy as np
from PIL import Image

# 文字画像表示
def ConvertToImg(img):
    return Image.fromarray(np.uint8(img))


# MNIST一文字の幅
chr_w = 28
# MNIST一文字の高さ
chr_h = 28
# 表示する文字数
num = math.floor(math.sqrt(len(false_pic_list)))

# MNISTの文字をPILで１枚の画像に描画する
canvas = Image.new('RGB', (int(chr_w * num/2), int(chr_h * num/2)), (255, 255, 255))

# MNISTの文字を読み込んで描画
i = 0
for y in range( int(num/2) ):
    for x in range( int(num/2) ):
        chrImg = ConvertToImg(false_pic[i].reshape(chr_w, chr_h))
        canvas.paste(chrImg, (chr_w*x, chr_h*y))
        i = i + 1
    

canvas.show()
# 表示した画像をJPEGとして保存
# canvas.save('mnist.jpg', 'JPEG', quality=100, optimize=True)