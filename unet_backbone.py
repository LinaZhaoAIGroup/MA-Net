import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import image as mpimg
from skimage.transform import resize
from tqdm import tqdm
import random
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import UpSampling2D,LeakyReLU,Input, Conv2D, BatchNormalization, Activation,MaxPool2D, MaxPooling2D, Dropout, Flatten, Dense,Conv2DTranspose,concatenate,Concatenate,ZeroPadding2D,add,Add
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential, load_model

from tensorflow import nn
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from skimage import io,data,color

# NO1.FCN-32S
# VGG16_FCN-32s
# 使用VGG16的前面13层卷积层，后面的3层fc层，也全换成卷积层
# 就是形如   def vgg16()： ...
#--------
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
import time
def rgb2gray(rgb):
#     r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
#     gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    return gray
def rgb2gray4d(rgb):
    r, g, b = rgb[:,:, :, 0], rgb[:,:, :, 1], rgb[:,:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
#     gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    return gray
def imgopen(imgpath,gray = 0):
    # open image and change to ndarray
    if gray == 0:
        img = Image.open(imgpath)  # PIL.Image打开并不是numpy数据 打开是原图
        npimg = np.array(img, dtype='float32')
    if gray == 1:
        img = Image.open(imgpath).convert("L")  # Image.open(x).convert("L")以灰度图形式打开
        npimg = np.array(img, dtype='float32')
    return npimg
def imgsave(img,savepath = "./??.png"):
    img = Image.fromarray(img.astype('uint8'))
    img.save(savepath)
    return 0
def sobelgamma(img):
    #传进来的肯定是 灰度图的ndarray
    h,w = img.shape
    #经典sobel 3x3 算子
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    d = np.zeros((h, w))
    dx = np.zeros((h, w))
    dy = np.zeros((h, w))
    Gx = np.zeros(d.shape)
    Gy = np.zeros(d.shape)
    #用numpy算的有点久 试试CV的
    for i in range(h - 2):
        for j in range(w - 2):
            Gx[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * sx))
            Gy[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * sy))
            # 最后的**0.5是做gamma变换了吧
            d[i + 1, j + 1] = (Gx[i + 1, j + 1] * Gx[i + 1, j + 1] + Gy[i + 1, j + 1] * Gy[i + 1, j + 1])
    # 不gamma这里出来效果跟电视雪花一样
    dgamma = d ** 0.5
    # imgsave(d, "./sobel.png")
    # imgsave(dgamma, "./sobelgamma.png")
    # imgsave(dx, "./dx.png")
    # imgsave(dy, "./dy.png")
    return dgamma
def cvsobel(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)

    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    dst = dst
    return dst

    # 原文链接：https: // blog.csdn.net / sunny2038 / article / details / 9170013
def medfilt(img):
    medimg = ndimage.median_filter(img, 3)#第二个参数是kernel size
    # imgsave(medimg,"./medimg.png")
    return medimg
def gsg(imgpath):
    # 存成numpy gray(中值滤波)+sobel+gamma-0.5
    # 新的三通道 1.gray 2.gray+soble_gamma-0.5 3.gray_medfilt+soble_gamma-0.5
    imgRGB = imgopen(imgpath,0)
    # gray = imgopen(imgpath,1)
    gray = rgb2gray(imgRGB)
    imgsave(gray,'./grayfomula1.png')
    # 第一个通道存灰度图
    c1 = gray
    # 第二个通道存gray_medfilt+soble_gamma-0.5
    c2med = medfilt(gray)
    c2 =  cvsobel(c2med)
    # 第三个通道存gray+soble_gamma-0.5
    c3 = cvsobel(gray)
    # imgsave(c1,"./c1.png")
    # imgsave(c2,"./cvsobelmedc2.png")
    # imgsave(c3,"./cvsobelc3.png")
    #要扩维再concat
    c1 = c1.reshape(c1.shape[0],c1.shape[1],1)
    c2 = c2.reshape(c2.shape[0],c2.shape[1],1)
    c3 = c3.reshape(c3.shape[0],c3.shape[1],1)
    c123= np.concatenate((c1,c2,c3),-1)
    channel6 = np.concatenate((imgRGB,c123),-1)

    return channel6
def gsg1(x):
    # 存成numpy gray(中值滤波)+sobel+gamma-0.5
    # 新的三通道 1.gray 2.gray+soble_gamma-0.5 3.gray_medfilt+soble_gamma-0.5
    # 这里接受的是 500,500,3的张量
    gray = rgb2gray(x)
    # 第一个通道存灰度图
    c1 = gray
    # 第二个通道存gray_medfilt+soble_gamma-0.5
    c2med = medfilt(gray)
    c2 =  cvsobel(c2med)
    # 第三个通道存gray+soble_gamma-0.5
    c3 = cvsobel(gray)
    # imgsave(c1,"./c1.png")
    # imgsave(c2,"./c2.png")
    # imgsave(c3,"./c3.png")
    #要扩维再concat
    c1 = c1.reshape(c1.shape[0],c1.shape[1],1)
    c2 = c2.reshape(c2.shape[0],c2.shape[1],1)
    c3 = c3.reshape(c3.shape[0],c3.shape[1],1)
    c123= np.concatenate((c1,c2,c3),-1)
    channel6 = np.concatenate((x,c123),-1)

    return channel6
def gRGB(x):
    #默认x 输进来的是3通道彩图
    gray = rgb2gray(x)
    # 第一个通道存灰度图
    r, g, b = x[:, :, 0], x[:, :, 1], x[:, :, 2] #这样的shape只有(x,y)没有channel的


    # imgsave(gray,"./ggg.png")
    # imgsave(r,"./r.png")
    # imgsave(g,"./g.png")
    # imgsave(b,"./b.png")
    #要扩维再concat
    gray = gray.reshape(gray.shape[0],gray.shape[1],1)
    r = r.reshape(r.shape[0],r.shape[1],1)
    g = g.reshape(g.shape[0],g.shape[1],1)
    b = b.reshape(b.shape[0],b.shape[1],1)
    # print(gray.shape)
    # print(r.shape)
    channel4 = np.concatenate((r,g,b,gray),-1)

    return channel4
def gRGB4d(x):
    #默认x 输进来的是3通道彩图
    gray = rgb2gray4d(x)
    # 第一个通道存灰度图
    r, g, b = x[:,:, :, 0], x[:,:, :, 1], x[:,:, :, 2] #这样的shape只有(x,y)没有channel的


    # imgsave(gray,"./ggg.png")
    # imgsave(r,"./r.png")
    # imgsave(g,"./g.png")
    # imgsave(b,"./b.png")
    #要扩维再concat
    gray = gray.reshape(gray.shape[0],gray.shape[1],gray.shape[2],1)
    r = r.reshape(r.shape[0],r.shape[1],r.shape[2],1)
    g = g.reshape(g.shape[0],g.shape[1],g.shape[2],1)
    b = b.reshape(b.shape[0],g.shape[1],g.shape[2],1)
    # print(gray.shape)
    # print(r.shape)
    channel4 = np.concatenate((r,g,b,gray),-1)
#--------
def fcn_vgg_blk(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def fcn_8s(input_img, n_filters = 64 ,dropout = 0.2 ,batchnorm = True):
    num_classes = 1
    #block 1
    c1 = fcn_vgg_blk(input_img, n_filters , kernel_size = 3 ,batchnorm = batchnorm)
    c2 = fcn_vgg_blk(c1, n_filters , kernel_size = 3 , batchnorm = batchnorm)
    p1 = MaxPool2D(pool_size = 2, strides=2, padding = 'valid')(c2)
    d1 = Dropout(dropout)(p1)
    
    #block 2
    c3 = fcn_vgg_blk(d1, n_filters*2 , kernel_size = 3 ,batchnorm = batchnorm)
    c4 = fcn_vgg_blk(c3, n_filters*2 , kernel_size = 3 , batchnorm = batchnorm)
    p2 = MaxPool2D(pool_size = 2, strides=2, padding = 'valid')(c4)
    d2 = Dropout(dropout)(p2)
    
    #block 3
    c5 = fcn_vgg_blk(d2, n_filters*4 , kernel_size = 3 ,batchnorm = batchnorm)
    c6 = fcn_vgg_blk(c5, n_filters*4 , kernel_size = 3 , batchnorm = batchnorm)
    c7 = fcn_vgg_blk(c6, n_filters*4 , kernel_size = 3 , batchnorm = batchnorm)
    p3 = MaxPool2D(pool_size = 2, strides=2, padding = 'valid')(c7)
    d3 = Dropout(dropout)(p3)
    
    #block 4
    c8 = fcn_vgg_blk(d3, n_filters*8 , kernel_size = 3 ,batchnorm = batchnorm)
    c9 = fcn_vgg_blk(c8, n_filters*8 , kernel_size = 3 , batchnorm = batchnorm)
    c10 = fcn_vgg_blk(c9, n_filters*8 , kernel_size = 3 , batchnorm = batchnorm)
    p4 = MaxPool2D(pool_size = 2, strides=2, padding = 'valid')(c10)
    d4 = Dropout(dropout)(p4)
    
    #block 5
    c11 = fcn_vgg_blk(d4, n_filters*8 , kernel_size = 3 ,batchnorm = batchnorm)
    c12 = fcn_vgg_blk(c11, n_filters*8 , kernel_size = 3 , batchnorm = batchnorm)
    c13 = fcn_vgg_blk(c12, n_filters*8 , kernel_size = 3 , batchnorm = batchnorm)
    p5 = MaxPool2D(pool_size = 2, strides=2, padding = 'valid')(c13)
    d5 = Dropout(dropout)(p5)
#     print("d5:",d5.shape)
    
    up4 = Conv2DTranspose(n_filters*8, kernel_size=(3, 3), strides=(2, 2), padding="same")(d5)
#     print("up4:",up4.shape)
    _16s = add([d4,up4])
    up3 = Conv2DTranspose(n_filters*4, kernel_size=(3, 3), strides=(2, 2), padding="same")(_16s)  
    _8s = add([d3,up3])
#     up1 = Conv2DTranspose(num_classes, kernel_size=(3, 3), strides=(8, 8), padding="same")(_8s)#返回去放大的倍数跟只跟stride有关
    up1 = UpSampling2D(size=(8, 8), interpolation="bilinear")(_8s)#这里csdn作者觉得 bilinear的UP 比 convtrans好
#     print("up1:",up1.shape)
    up1 = fcn_vgg_blk(up1, n_filters , kernel_size = 3 , batchnorm = batchnorm)
#     print("up1:",up1.shape)
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (up1)
    model = Model(inputs = [input_img],outputs=[outputs])
    return model

def fcn_vgg16_func(input_img, n_filters=16, dropout=0.2, batchnorm=True):  #n_filters 64->16
    num_classes = 1
    # block 1
    c1 = fcn_vgg_blk(input_img, n_filters, kernel_size=3, batchnorm=batchnorm)
    c2 = fcn_vgg_blk(c1, n_filters, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPool2D(pool_size=2, strides=2, padding='valid')(c2)
    d1 = Dropout(dropout)(p1)

    # block 2
    c3 = fcn_vgg_blk(d1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    c4 = fcn_vgg_blk(c3, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPool2D(pool_size=2, strides=2, padding='valid')(c4)
    d2 = Dropout(dropout)(p2)

    # block 3
    c5 = fcn_vgg_blk(d2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    c6 = fcn_vgg_blk(c5, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    c7 = fcn_vgg_blk(c6, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPool2D(pool_size=2, strides=2, padding='valid')(c7)
    d3 = Dropout(dropout)(p3)

    # block 4
    c8 = fcn_vgg_blk(d3, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    c9 = fcn_vgg_blk(c8, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    c10 = fcn_vgg_blk(c9, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPool2D(pool_size=2, strides=2, padding='valid')(c10)
    d4 = Dropout(dropout)(p4)

    # block 5
    c11 = fcn_vgg_blk(d4, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    c12 = fcn_vgg_blk(c11, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    c13 = fcn_vgg_blk(c12, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p5 = MaxPool2D(pool_size=2, strides=2, padding='valid')(c13)
    d5 = Dropout(dropout)(p5)

    #     #block 6 fully connected layers
    #     # fcn32s block 6 use convolution layers instead of fc layers
    c14 = fcn_vgg_blk(d5, 4096, kernel_size=7, batchnorm=batchnorm) #4096  跑不通  换2048
    d6 = Dropout(dropout)(c14)  
    c15 = fcn_vgg_blk(d6, 4096, kernel_size=1, batchnorm=batchnorm) #4096 跑不通  换2048
    d7 = Dropout(dropout)(c15)

    #     #classifying layer
    c16 = fcn_vgg_blk(d7, num_classes, kernel_size=1, batchnorm=batchnorm)
    # #     # upsampling
    #     c17 = Conv2DTranspose(filters=num_classes, kernel_size=(32, 32), strides=(32, 32), padding="valid",activation='sigmoid')(c16)

    c17 = Conv2DTranspose(filters=num_classes, kernel_size=(32, 32), strides=(32, 32), padding="valid")(c16)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c17)
#     outputs = c17
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
# #test on fcn VGGwithFuncAPI
# input_img = Input((224, 224, 3),name="imgtest")
# model = fcn_vgg16_func(input_img)
# model.summary(line_length=120,positions=[0.40,0.60,0.7,1.])
# x = np.random.randn(1,224,224,3)
# out = model(x)
# print(out.shape

# NO2.attention_Unet
def conv_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        # BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


# test
# test = np.random.rand(1,224,224,3)
# test = conv_block(test,64)
# print(test.shape) #(1, 224, 224, 64)

def up_conv(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    # keras.layers.convolutional.UpSampling2D(size=(2, 2), data_format=None)
    x = UpSampling2D(size=(2, 2))(input_tensor)

    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)

    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


# test
# test = np.random.rand(1,224,224,3)
# test = up_conv(test,64)
# print(test.shape) #(1, 448, 448, 64)


def Attention_block(inputg, inputx, F_int):
    #     print("INPUTX , inputg",inputx.shape,inputg.shape)
    W_g = Conv2D(filters=F_int, kernel_size=(1, 1), kernel_initializer="he_normal", padding="valid")(inputg)
    W_g = BatchNormalization()(W_g)
    #     print("W_g:",W_g.shape)
    W_x = Conv2D(filters=F_int, kernel_size=(1, 1), kernel_initializer="he_normal", padding="valid")(inputx)
    W_x = BatchNormalization()(W_x)
    #     print("W_x:",W_x.shape)
    # concat + relu
    psi = W_g + W_x
    psi = Activation("relu")(psi)
    psi = Conv2D(filters=1, kernel_size=(1, 1), kernel_initializer="he_normal", padding="valid")(psi)
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)
    #     print("psi:",psi.shape)
    psi = inputx * psi  # 这里是对应元素相乘 不是矩阵乘
    #     print("psi * x:",psi.shape)
    return psi


# test
# testg= np.array(np.random.rand(1,224,224,32),dtype = np.float32)
# testx= np.array(np.random.rand(1,224,224,32),dtype = np.float32)

# test = Attention_block(testg,testx,64)
# print(test.shape)
# W_g: (1, 224, 224, 64)
# W_x: (1, 224, 224, 64)
# psi: (1, 224, 224, 1)
# psi * x: (1, 224, 224, 32)
# (1, 224, 224, 32)

def AttU_Net(input_img, n_filters=16, dropout=0.2, batchnorm=True):
    # print("ori_img:", input_img.shape)
    n_filters = [n_filters, n_filters * 2, n_filters * 4, n_filters * 8, n_filters * 16]

    # encoding path
    c1 = conv_block(input_img, n_filters[0], kernel_size=3, batchnorm=batchnorm)
    # print("c1:", c1.shape)
    p1 = MaxPooling2D((2, 2))(c1)
    # print("p1:", p1.shape)
    p1 = Dropout(dropout)(p1)

    c2 = conv_block(p1, n_filters[1], kernel_size=3, batchnorm=batchnorm)
    # print("c2:", c2.shape)
    p2 = MaxPooling2D((2, 2))(c2)
    # print("p2:", p2.shape)
    p2 = Dropout(dropout)(p2)

    c3 = conv_block(p2, n_filters[2], kernel_size=3, batchnorm=batchnorm)
    # print("c3:", c3.shape)
    p3 = MaxPooling2D((2, 2))(c3)
    # print("p3:", p3.shape)
    p3 = Dropout(dropout)(p3)

    c4 = conv_block(p3, n_filters[3], kernel_size=3, batchnorm=batchnorm)
    # print("c4:", c4.shape)
    p4 = MaxPooling2D((2, 2))(c4)
    # print("p4:", p4.shape)
    p4 = Dropout(dropout)(p4)

    c5 = conv_block(p4, n_filters[4], kernel_size=3, batchnorm=batchnorm)
    # print("c5:", c5.shape)

    # decoding + concat path
    u5 = up_conv(c5, n_filters[3], kernel_size=3, batchnorm=batchnorm)
    # print("u5:", u5.shape)
    att4 = Attention_block(inputg=u5, inputx=c4, F_int=n_filters[2])
    # print("att4:", att4.shape)
    u5 = concatenate([att4, u5])
    # print("cat u5 att4:", u5.shape)
    u5 = conv_block(u5, n_filters[3], kernel_size=3, batchnorm=batchnorm)
    # print("u5:", u5.shape)

    u4 = up_conv(u5, n_filters[2], kernel_size=3, batchnorm=batchnorm)
    # print("u4:", u4.shape)
    att3 = Attention_block(inputg=u4, inputx=c3, F_int=n_filters[1])
    # print("att3:", att3.shape)
    u4 = concatenate([att3, u4])
    # print("cat u4 att3:", u4.shape)
    u4 = conv_block(u4, n_filters[2], kernel_size=3, batchnorm=batchnorm)
    # print("u4:", u4.shape)

    u3 = up_conv(u4, n_filters[1], kernel_size=3, batchnorm=batchnorm)
    # print("u3:", u3.shape)
    att2 = Attention_block(inputg=u3, inputx=c2, F_int=n_filters[0])
    # print("att2:", att2.shape)
    u3 = concatenate([att2, u3])
    # print("cat u3 att2:", u3.shape)
    u3 = conv_block(u3, n_filters[1], kernel_size=3, batchnorm=batchnorm)
    # print("u3:", u3.shape)

    u2 = up_conv(u3, n_filters[0], kernel_size=3, batchnorm=batchnorm)
    # print("u2:", u2.shape)
    att1 = Attention_block(inputg=u2, inputx=c1, F_int=int(n_filters[0] / 2))
    # print("att1:", att1.shape)
    u2 = concatenate([att1, u2])
    # print("cat u2 att1:", u2.shape)
    u2 = conv_block(u2, n_filters[0], kernel_size=3, batchnorm=batchnorm)
    # print("u2:", u2.shape)

    u1 = Conv2D(filters=1, kernel_size=(1, 1), kernel_initializer="he_normal", padding="valid")(u2)
    # print("u1:", u1.shape)
    u1 = Activation('sigmoid')(u1)

    outputs = u1

    model = Model(inputs=[input_img], outputs=[outputs])
    return model
# #test
# input_img = Input((224, 224, 3),name="imgtest")
# model = AttU_Net(input_img)
# model.summary(line_length=120,positions=[0.40,0.60,0.7,1.])
# print("model layers :",len(model.layers))
# x = np.random.randn(1,224,224,3)
# out = model(x)
# print(out.shape)
# NO3 SegNet
def MaxPoolingWithArgmax2D(inputs, kernel_size=2, stride=2, padding='same'):
    output, argmax = tf.nn.max_pool_with_argmax(
        inputs,
        ksize=[1, kernel_size, kernel_size, 1],
        strides=[1, stride, stride, 1],
        padding=padding.upper())
    # 转为float类型
    argmax = K.cast(argmax, K.floatx())
    return [output, argmax]


def MaxUnpooling2D(inputs, size=(2, 2), output_shape=None):
    updates, mask = inputs[0], inputs[1]
    mask = K.cast(mask, 'int32')
    input_shape = tf.shape(updates, out_type='int32')

    if output_shape is None:
        output_shape = (
            input_shape[0],
            input_shape[1] * size[0],
            input_shape[2] * size[1],
            input_shape[3])

    ret = tf.scatter_nd(K.expand_dims(K.flatten(mask)),
                        K.flatten(updates),
                        [K.prod(output_shape)])

    input_shape = updates.shape
    out_shape = [-1,
                 input_shape[1] * size[0],
                 input_shape[2] * size[1],
                 input_shape[3]]
    return K.reshape(ret, out_shape)


def segnet_conv_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        # BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def Segnet(input_img, n_filters=16, dropout=0.2, batchnorm=True):
    # print("ori_img:", input_img.shape)
    n_filters = [n_filters, n_filters * 2, n_filters * 4, n_filters * 8, n_filters * 16]

    # encoding path
    x11 = segnet_conv_block(input_img, n_filters[0], kernel_size=3, batchnorm=batchnorm)
    # print("x11:", x11.shape)
    x12 = segnet_conv_block(x11, n_filters[0], kernel_size=3, batchnorm=batchnorm)
    # print("x12:", x12.shape)
    x1p, id1 = MaxPoolingWithArgmax2D(x12)
    # print("x1p:", x1p.shape, "id1:", id1.shape)
    x1p = Dropout(dropout)(x1p)  # 只对要学习的部分进行dropout

    x21 = segnet_conv_block(x1p, n_filters[1], kernel_size=3, batchnorm=batchnorm)
    # print("x21:", x21.shape)
    x22 = segnet_conv_block(x21, n_filters[1], kernel_size=3, batchnorm=batchnorm)
    # print("x22:", x22.shape)
    x2p, id2 = MaxPoolingWithArgmax2D(x22)
    # print("x2p:", x2p.shape, "id2:", id2.shape)
    x2p = Dropout(dropout)(x2p)  # 只对要学习的部分进行dropout

    x31 = segnet_conv_block(x2p, n_filters[2], kernel_size=3, batchnorm=batchnorm)
    # print("x31:", x31.shape)
    x32 = segnet_conv_block(x31, n_filters[2], kernel_size=3, batchnorm=batchnorm)
    # print("x32:", x32.shape)
    x33 = segnet_conv_block(x32, n_filters[2], kernel_size=3, batchnorm=batchnorm)
    # print("x33:", x33.shape)
    x3p, id3 = MaxPoolingWithArgmax2D(x33)
    # print("x3p:", x3p.shape, "id3:", id3.shape)
    x3p = Dropout(dropout)(x3p)  # 只对要学习的部分进行dropout

    x41 = segnet_conv_block(x3p, n_filters[3], kernel_size=3, batchnorm=batchnorm)
    # print("x41:", x41.shape)
    x42 = segnet_conv_block(x41, n_filters[3], kernel_size=3, batchnorm=batchnorm)
    # print("x42:", x42.shape)
    x43 = segnet_conv_block(x42, n_filters[3], kernel_size=3, batchnorm=batchnorm)
    # print("x43:", x43.shape)
    x4p, id4 = MaxPoolingWithArgmax2D(x43)
    # print("x4p:", x4p.shape, "id4:", id4.shape)
    x4p = Dropout(dropout)(x4p)  # 只对要学习的部分进行dropout

    x51 = segnet_conv_block(x4p, n_filters[3], kernel_size=3, batchnorm=batchnorm)
    # print("x51:", x51.shape)
    x52 = segnet_conv_block(x51, n_filters[3], kernel_size=3, batchnorm=batchnorm)
    # print("x52:", x52.shape)
    x53 = segnet_conv_block(x52, n_filters[3], kernel_size=3, batchnorm=batchnorm)
    # print("x53:", x53.shape)
    x5p, id5 = MaxPoolingWithArgmax2D(x53)
    # print("x5p:", x5p.shape, "id5:", id5.shape)
    x5p = Dropout(dropout)(x5p)  # 只对要学习的部分进行dropout

    # decoding path
    x5d = MaxUnpooling2D([x5p, id5])
    # print("x5d:", x5d.shape)
    x53d = segnet_conv_block(x5d, n_filters[3], kernel_size=3, batchnorm=batchnorm)
    # print("x53d:", x53d.shape)
    x52d = segnet_conv_block(x53d, n_filters[3], kernel_size=3, batchnorm=batchnorm)
    # print("x52d:", x52d.shape)
    x51d = segnet_conv_block(x52d, n_filters[3], kernel_size=3, batchnorm=batchnorm)
    # print("x51d:", x51d.shape)

    x4d = MaxUnpooling2D([x51d, id4])
    # print("x4d:", x4d.shape)
    x43d = segnet_conv_block(x4d, n_filters[3], kernel_size=3, batchnorm=batchnorm)
    # print("x43d:", x43d.shape)
    x42d = segnet_conv_block(x43d, n_filters[3], kernel_size=3, batchnorm=batchnorm)
    # print("x42d:", x42d.shape)
    x41d = segnet_conv_block(x42d, n_filters[2], kernel_size=3, batchnorm=batchnorm)
    # print("x41d:", x41d.shape)

    x3d = MaxUnpooling2D([x41d, id3])
    # print("x3d:", x3d.shape)
    x33d = segnet_conv_block(x3d, n_filters[2], kernel_size=3, batchnorm=batchnorm)
    # print("x33d:", x33d.shape)
    x32d = segnet_conv_block(x33d, n_filters[2], kernel_size=3, batchnorm=batchnorm)
    # print("x32d:", x32d.shape)
    x31d = segnet_conv_block(x32d, n_filters[1], kernel_size=3, batchnorm=batchnorm)
    # print("x31d:", x31d.shape)

    x2d = MaxUnpooling2D([x31d, id2])
    # print("x2d:", x2d.shape)
    x22d = segnet_conv_block(x2d, n_filters[1], kernel_size=3, batchnorm=batchnorm)
    # print("x22d:", x22d.shape)
    x21d = segnet_conv_block(x22d, n_filters[0], kernel_size=3, batchnorm=batchnorm)
    # print("x21d:", x21d.shape)

    x1d = MaxUnpooling2D([x21d, id1])
    # print("x1d:", x1d.shape)
    x12d = segnet_conv_block(x1d, n_filters[0], kernel_size=3, batchnorm=batchnorm)
    # print("x12d:", x12d.shape)
    x11d = segnet_conv_block(x12d, 1, kernel_size=3, batchnorm=batchnorm)
    # print("x11d:", x11d.shape)

    #     u1 = Conv2D(filters=1, kernel_size=(1, 1), kernel_initializer="he_normal",padding="valid")(u2)
    #     print("u1:",u1.shape)
    #     u1 = Activation('sigmoid')(u1)

    out = Activation('sigmoid')(x11d)

    outputs = out

    model = Model(inputs=[input_img], outputs=[outputs])
    return model
# #test
# input_img = Input((224, 224, 3),name="imgtest")
# model = Segnet(input_img)
# model.summary(line_length=120,positions=[0.40,0.60,0.7,1.])
# print("model layers :",len(model.layers))
# x = np.random.randn(1,224,224,3)
# out = model(x)
# print(out.shape)

# NO4.CENet
def cenet_conv_block(input_tensor, n_filters, kernel_size=3, strides=1, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), strides=(strides, strides),
               kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        # BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def DACblock(input_tensor, n_filters, kernel_size=3, padding='same', dilation_rate=1, batchnorm=True):
    x11 = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                 padding=padding, dilation_rate=1)(input_tensor)
    if batchnorm:
        x1 = BatchNormalization()(x11)
    x11 = Activation("relu")(x11)
    #     print("x11.shape:",x11.shape)

    x21 = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                 padding=padding, dilation_rate=3)(input_tensor)
    x22 = Conv2D(filters=n_filters, kernel_size=(1, 1), kernel_initializer="he_normal",
                 padding=padding, dilation_rate=1)(x21)
    if batchnorm:
        x22 = BatchNormalization()(x22)
    x22 = Activation("relu")(x22)
    #     print("x22.shape:",x22.shape)

    x31 = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                 padding=padding, dilation_rate=1)(input_tensor)
    x32 = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                 padding=padding, dilation_rate=3)(x31)
    x33 = Conv2D(filters=n_filters, kernel_size=(1, 1), kernel_initializer="he_normal",
                 padding=padding, dilation_rate=1)(x32)
    if batchnorm:
        x33 = BatchNormalization()(x33)
    x33 = Activation("relu")(x33)
    #     print("x33.shape:",x33.shape)

    x41 = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                 padding=padding, dilation_rate=1)(input_tensor)
    x42 = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                 padding=padding, dilation_rate=3)(x41)
    x43 = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                 padding=padding, dilation_rate=5)(x42)
    x44 = Conv2D(filters=n_filters, kernel_size=(1, 1), kernel_initializer="he_normal",
                 padding=padding, dilation_rate=1)(x43)
    if batchnorm:
        x44 = BatchNormalization()(x44)
    x44 = Activation("relu")(x44)
    #     print("x44.shape:",x44.shape)

    # 因为这里要简单的对应元素相加 所以他们的维度一定是相同的
    x = input_tensor + x11 + x22 + x33 + x44
    return x
# #test
# test = np.random.rand(1,224,224,3)
# test = DACblock(test,3)
# print(test.shape) #(1, 224, 224, 64)
def SPPblock(input_tensor, n_filters=1, kernel_size=1, batchnorm=True):
    # size 512 到这边 是 16   size 448到这里是 14
    # 然后这边 最后都是1通道
    # upsampling 可以直接理解为 pooling的反向操作  不需要参数的学习  conv2dtranspose才是需要参数学习的
    # 这个模块就是先进行 maxpool 然后 再upsamp 就完事了
    # 20210602
    pool1 = MaxPool2D(pool_size=(2, 2), strides=2)(input_tensor)
    #     print("up1:",pool1.shape)
    conv1 = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                   padding="same")(pool1)
    up1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv1)
    #     print("up1:",up1.shape)

    # 16-zeropadding-18-maxpool-6-upsamp-18-maxpool-16
    pool2 = ZeroPadding2D(padding=(1, 1), data_format=None)(input_tensor)
    #     print("up2:",pool2.shape)
    pool2 = MaxPool2D(pool_size=(3, 3), strides=3)(pool2)
    #     print("up2:",pool2.shape)
    conv2 = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                   padding="same")(pool2)
    up2 = UpSampling2D(size=(3, 3), interpolation='bilinear')(conv2)
    #     print("up2:",up2.shape)
    up2 = MaxPool2D(pool_size=(3, 3), strides=1)(up2)
    #     print("up2:",up2.shape)

    # 16-zeropadding-20-maxpool-4-upsamp-20-maxpool-16
    pool3 = ZeroPadding2D(padding=(2, 2), data_format=None)(input_tensor)
    #     print("up3:",pool3.shape)
    pool3 = MaxPool2D(pool_size=(5, 5), strides=5)(pool3)
    #     print("up3:",pool3.shape)
    conv3 = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                   padding="same")(pool3)
    up3 = UpSampling2D(size=(5, 5), interpolation='bilinear')(conv3)
    #     print("up3:",up3.shape)
    up3 = MaxPool2D(pool_size=(5, 5), strides=1)(up3)
    #     print("up3:",up3.shape)

    # 16-zeropadding-18-maxpool-3-upsamp-18-maxpool-16
    pool4 = ZeroPadding2D(padding=(1, 1), data_format=None)(input_tensor)
    #     print("up4:",pool4.shape)
    pool4 = MaxPool2D(pool_size=(6, 6), strides=6)(pool4)
    #     print("up4:",pool4.shape)
    conv4 = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                   padding="same")(pool4)
    up4 = UpSampling2D(size=(6, 6), interpolation='bilinear')(conv4)
    #     print("up4:",up4.shape)
    up4 = MaxPool2D(pool_size=(3, 3), strides=1)(up4)
    #     print("up4:",up4.shape)

    out = concatenate([up1, up2, up3, up4, input_tensor])
    return out
# #test
# test = np.random.rand(1,16,16,3)
# test = SPPblock(test)
# print(test.shape) #(1, 224, 224, 64)
def DecoderBlock(input_tensor, n_filters=16, kernel_size=3, batchnorm=True):
    #     Conv2DTranspose(filters = nb_filter[0],kernel_size= (2, 2), strides=(2, 2), name='up01', padding='same')
    filters = input_tensor.shape[-1] // 4
    #     filters = n_filters // 4
    conv1 = cenet_conv_block(input_tensor, filters, kernel_size=1, batchnorm=batchnorm)
    #     print("conv1:",conv1.shape)

    deconv2 = Conv2DTranspose(filters=filters, kernel_size=3, strides=(2, 2), padding='same')(conv1)
    deconv2 = BatchNormalization()(deconv2)
    deconv2 = Activation("relu")(deconv2)
    #     print("deconv2:",deconv2.shape)

    conv3 = cenet_conv_block(deconv2, n_filters, kernel_size=1, batchnorm=batchnorm)
    #     print("conv3:",conv3.shape)

    out = conv3
    return out
# #test
# test = np.random.rand(1,512,512,16)
# test = DecoderBlock(test)
# print(test.shape) #(1, 224, 224, 64)
def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x

#对于identity_Block的nb_filter以及bottleneck_Block的nb_filters进行说明：如上图所示：resNet34里面，每一个卷积层当中的nb_filter仅仅是一种，而resnet50里面每一层的filter有多种
def identity_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:#shortcut的含义是：将输入层x与最后的输出层y进行连接，如上图所示
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x
def CENet(input_img, n_filters = 16 ,dropout = 0.2 ):
    n_filters = [n_filters, n_filters*2 , n_filters*4, n_filters*8, n_filters*16]
    x = input_img
    # print("conv0:",x.shape)
    #-----------------------------------用了resnet34的部分--CE-net的encoder---------------------
    #conv1
#     x = Conv2d_BN(x, nb_filter=n_filters[0], kernel_size=(7, 7), strides=(2, 2), padding='same')
    e0 = cenet_conv_block(x,n_filters[0], kernel_size=7,strides=2 ,batchnorm=True)
    # print("conv0e0:",e0.shape)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(e0)
    # print("conv1:",x.shape)
    #conv2_x
    x = identity_Block(x, nb_filter=n_filters[0], kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=n_filters[0], kernel_size=(3, 3))
    e1 = identity_Block(x, nb_filter=n_filters[0], kernel_size=(3, 3))
    # print("conv2e1:",e1.shape)
    #conv3_x
    x = identity_Block(e1, nb_filter=n_filters[1], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=n_filters[1], kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=n_filters[1], kernel_size=(3, 3))
    e2 = identity_Block(x, nb_filter=n_filters[1], kernel_size=(3, 3))
    # print("conv3e2:",e2.shape)
    #conv4_x
    x = identity_Block(e2, nb_filter=n_filters[2], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=n_filters[2], kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=n_filters[2], kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=n_filters[2], kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=n_filters[2], kernel_size=(3, 3))
    e3 = identity_Block(x, nb_filter=n_filters[2], kernel_size=(3, 3))
    # print("conv4e3:",e3.shape)
    #conv5_x
    x = identity_Block(e3, nb_filter=n_filters[3], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=n_filters[3], kernel_size=(3, 3))
    e4 = identity_Block(x, nb_filter=n_filters[3], kernel_size=(3, 3))
    # print("conv5e4:",e4.shape)
    #-----------------------------------至此用了resnet34的部分--前面属于CE-net的encoder---------------------
    #现在进入----------------------cenet的center部分--------------------------------------------------------
    e4 = DACblock(e4,n_filters[3])
    # print("dac-e4:",e4.shape)
    e4 = SPPblock(e4)
    # print("spp-e4:",e4.shape)
    #----------------------the end--------------------------------------------------------------------------
    #现在进入----------------------cenet的deconder部分--------------------------------------------------------
    d4 = DecoderBlock(e4,n_filters[2]) + e3
    # print("d4:",d4.shape)
    d3 = DecoderBlock(d4,n_filters[1]) + e2
    # print("d3:",d3.shape)
    d2 = DecoderBlock(d3,n_filters[0]) + e1
    # print("d2:",d2.shape)
    d1 = DecoderBlock(d2,n_filters[0]) + e0
    # print("d1:",d1.shape)
    #----------------------the end--------------------------------------------------------------------------
    #----------------------------Final----------------------------------------------------------------------
    end1 = Conv2DTranspose(filters = n_filters[0]//2 ,kernel_size= 4, strides=(2, 2), padding='same')(d1)
    end1 = Activation('relu')(end1)
    # print("end1:",end1.shape)
    end2 = Conv2D(n_filters[0]//2, kernel_size = 3 , padding="same")(end1)
    end2 = Activation('relu')(end2)
    # print("end2:",end2.shape)
    end3 = Conv2D(1, kernel_size = 3 , padding="same")(end2)
    end3 = Activation('sigmoid')(end3)
    # print("end3:",end3.shape)
    #----------------------the end--------------------------------------------------------------------------
    outputs = end3
    model = Model(inputs = [input_img],outputs=[outputs])
    return model
# input_img = Input((512, 512, 3),name="imgtest")
# model = CENet(input_img)
# model.summary(line_length=120,positions=[0.40,0.60,0.7,1.])
# print("model layers :",len(model.layers))
# x = np.random.randn(1,448,448,3)
# out = model(x)
# print(out.shape)

# NO5 NestedUNet Unet++ 
def Unetpp(input_img, n_filters=16, dropout=0.2, batchnorm=True):
    
    n_filters = [n_filters, n_filters * 2, n_filters * 4, n_filters * 8, n_filters * 16]
    
    x00 = conv_block(input_img, n_filters[0], kernel_size=3, batchnorm=batchnorm)
#     print("x00:",x00.shape)
    p00 = MaxPooling2D((2, 2))(x00)
    p00 = Dropout(dropout)(p00)
#     print("p00:",p00.shape)
    
    x10 = conv_block(p00, n_filters[1], kernel_size=3, batchnorm=batchnorm)
#     print("x10:",x10.shape)
    up01 = Conv2DTranspose(filters=n_filters[0], kernel_size=(2, 2), strides=(2, 2), name='up01', padding='same')(x10)
#     print("up01:",up01.shape)
    # t c  =  transport  and  contatenate 到01这个node上
    tc01 = concatenate([x00,up01])
    tc01 = Dropout(dropout)(tc01)
#     print("tc01:",tc01.shape)
    x01 =  conv_block(tc01, n_filters[0], kernel_size=3, batchnorm=batchnorm)
#     print("x01:",x01.shape)
    
    p10 = MaxPooling2D((2, 2))(x10)
    p10 = Dropout(dropout)(p10)
#     print("p10:",p10.shape)
    x20 = conv_block(p10, n_filters[2], kernel_size=3, batchnorm=batchnorm)
#     print("x20:",x20.shape)
    up11 = Conv2DTranspose(filters=n_filters[1], kernel_size=(2, 2), strides=(2, 2), name='up11',padding='same')(x20)
#     print("up11:",up11.shape)
    tc11 = concatenate([x10,up11])
    tc11 = Dropout(dropout)(tc11)
#     print("tc11:",tc11.shape)
    x11 =  conv_block(tc11, n_filters[1], kernel_size=3, batchnorm=batchnorm)
#     print("x11:",x11.shape)
    
    up02 = Conv2DTranspose(filters=n_filters[0], kernel_size=(2, 2), strides=(2, 2), name='up02',padding='same')(x11)
#     print("up02:",up02.shape)
    tc02 = concatenate([x00,x01,up02])
    tc02 = Dropout(dropout)(tc02)
#     print("tc02:",tc02.shape)
    x02 =  conv_block(tc02, n_filters[0], kernel_size=3, batchnorm=batchnorm)
#     print("x02:",x02.shape)
    
    p20 = MaxPooling2D((2, 2))(x20)
    p20 = Dropout(dropout)(p20)
#     print("p20:",p20.shape)
    x30 = conv_block(p20, n_filters[3], kernel_size=3, batchnorm=batchnorm)
#     print("x30:",x30.shape)    
    up21 = Conv2DTranspose(filters=n_filters[2], kernel_size=(2, 2), strides=(2, 2), name='up21',padding='same')(x30)
#     print("up21:",up21.shape)
    tc21 = concatenate([x20,up21])
    tc21 = Dropout(dropout)(tc21)
#     print("tc21:",tc21.shape)
    x21 =  conv_block(tc21, n_filters[2], kernel_size=3, batchnorm=batchnorm)
#     print("x21:",x21.shape)
        
    up12 = Conv2DTranspose(filters=n_filters[1], kernel_size=(2, 2), strides=(2, 2), name='up12',padding='same')(x21)
#     print("up12:",up12.shape)
    tc12 = concatenate([x10, x11, up12])
    tc12 = Dropout(dropout)(tc12)
#     print("tc12:",tc12.shape)
    x12 =  conv_block(tc12, n_filters[1], kernel_size=3, batchnorm=batchnorm)
#     print("x12:",x12.shape)
        
    up03 = Conv2DTranspose(filters=n_filters[0], kernel_size=(2, 2), strides=(2, 2), name='up03',padding='same')(x12)
#     print("up03:",up03.shape)
    tc03 = concatenate([x00, x01, x02, up03])
    tc03 = Dropout(dropout)(tc03)
#     print("tc03:",tc03.shape)
    x03 =  conv_block(tc03, n_filters[0], kernel_size=3, batchnorm=batchnorm)
#     print("x03:",x03.shape)
    
    p30 = MaxPooling2D((2, 2))(x30)
    p30 = Dropout(dropout)(p30)
#     print("p30:",p30.shape)
    x40 = conv_block(p30, n_filters[4], kernel_size=3, batchnorm=batchnorm)
#     print("x40:",x40.shape)   
    
    up31 = Conv2DTranspose(filters=n_filters[3], kernel_size=(2, 2), strides=(2, 2), name='up31',padding='same')(x40)
#     print("up31:",up31.shape)
    tc31 = concatenate([x30,up31])
    tc31 = Dropout(dropout)(tc31)
#     print("tc31:",tc31.shape)
    x31 =  conv_block(tc31, n_filters[3], kernel_size=3, batchnorm=batchnorm)
#     print("x31:",x31.shape)
        
    up22 = Conv2DTranspose(filters=n_filters[2], kernel_size=(2, 2), strides=(2, 2), name='up22',padding='same')(x31)
#     print("up22:",up22.shape)
    tc22 = concatenate([x20, x21, up22])
    tc22 = Dropout(dropout)(tc22)
#     print("tc22:",tc22.shape)
    x22 =  conv_block(tc22, n_filters[2], kernel_size=3, batchnorm=batchnorm)
#     print("x22:",x22.shape)
        
    up13 = Conv2DTranspose(filters=n_filters[1], kernel_size=(2, 2), strides=(2, 2), name='up13',padding='same')(x22)
#     print("up13:",up13.shape)
    tc13 = concatenate([x10, x11, x12, up13])
    tc13 = Dropout(dropout)(tc13)
#     print("tc13:",tc13.shape)
    x13 =  conv_block(tc13, n_filters[1], kernel_size=3, batchnorm=batchnorm)
#     print("x13:",x13.shape)
    
    up04 = Conv2DTranspose(filters=n_filters[0], kernel_size=(2, 2), strides=(2, 2), name='up04',padding='same')(x13)
#     print("up04:",up04.shape)
    tc04 = concatenate([x00, x01, x02, x03 ,up04])
    tc04 = Dropout(dropout)(tc04)
#     print("tc04:",tc04.shape)
    x04 =  conv_block(tc04, n_filters[0], kernel_size=3, batchnorm=batchnorm)
#     print("x04:",x04.shape)
    
    out = Conv2D(1, 1)(x04)
#     print("out:",out.shape)
    out = Activation('sigmoid')(out)
#     print("out:",out.shape)
    outputs = out
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
class DoubleConv2d(Model):
    # 这个C包括了Cba了
    def __init__(self, filters):
        super(DoubleConv2d, self).__init__()
        # 卷积就是特征提取器 CBAPD  convolution  batchnormalization activation pooling dropout
        self.filters = filters
        self.c1 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.b2 = BatchNormalization()
        self.a2 = Activation('relu')

    # 这里必须是 call
    def call(self, x):
        x = self.c1(x)
        #         print(x.shape)
        x = self.b1(x)
        #         print(x.shape)
        x = self.a1(x)
        #         print(x.shape)

        x = self.c2(x)
        #         print(x.shape)
        x = self.b2(x)
        #         print(x.shape)
        y = self.a2(x)
        #         print(y.shape)

        return y


base = 8  # unet64我的电脑肯定跑不动   #unetpp 16我电脑跑不动
# [64,128,256,512,1024]    [16,32,64,128,256]
nb_filter = [base * 1, base * 2, base * 4, base * 8, base * 16]
# 搞一下dropout
dropoutrate = 0.25


class NestedUNet(Model):
    def __init__(self):
        super(NestedUNet, self).__init__()

        #         self.args = args
        self.args = None  # 这里是用来剪枝的 先不管
        self.dropout = True
        self.print = True

        #         nb_filter = [32, 64, 128, 256, 512]

        #         self.pool = nn.MaxPool2d(2, 2)
        self.pool = MaxPool2D(pool_size=(2, 2), strides=2)  # 池化层
        # CLASS torch.nn.Upsample(size=None, scale_factor=None, mode='nearest', align_corners=None)
        # scale_factor=？ 指定输出为输入的
        # 最原始版本这里用  Conv2dTranspose 代替这里的 Upsample
        # up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)

        #         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.up0_1 = Conv2DTranspose(filters=nb_filter[0], kernel_size=(2, 2), strides=(2, 2), name='up01',
                                     padding='same')
        self.up1_1 = Conv2DTranspose(filters=nb_filter[1], kernel_size=(2, 2), strides=(2, 2), name='up11',
                                     padding='same')
        self.up0_2 = Conv2DTranspose(filters=nb_filter[0], kernel_size=(2, 2), strides=(2, 2), name='up02',
                                     padding='same')
        self.up2_1 = Conv2DTranspose(filters=nb_filter[2], kernel_size=(2, 2), strides=(2, 2), name='up21',
                                     padding='same')
        self.up1_2 = Conv2DTranspose(filters=nb_filter[1], kernel_size=(2, 2), strides=(2, 2), name='up12',
                                     padding='same')
        self.up0_3 = Conv2DTranspose(filters=nb_filter[0], kernel_size=(2, 2), strides=(2, 2), name='up03',
                                     padding='same')
        self.up3_1 = Conv2DTranspose(filters=nb_filter[3], kernel_size=(2, 2), strides=(2, 2), name='up31',
                                     padding='same')
        self.up2_2 = Conv2DTranspose(filters=nb_filter[2], kernel_size=(2, 2), strides=(2, 2), name='up22',
                                     padding='same')
        self.up1_3 = Conv2DTranspose(filters=nb_filter[1], kernel_size=(2, 2), strides=(2, 2), name='up13',
                                     padding='same')
        self.up0_4 = Conv2DTranspose(filters=nb_filter[0], kernel_size=(2, 2), strides=(2, 2), name='up04',
                                     padding='same')

        self.conv0_0 = DoubleConv2d(nb_filter[0])
        self.conv1_0 = DoubleConv2d(nb_filter[1])
        self.conv2_0 = DoubleConv2d(nb_filter[2])
        self.conv3_0 = DoubleConv2d(nb_filter[3])
        self.conv4_0 = DoubleConv2d(nb_filter[4])

        #         self.conv0_1 = DoubleConv(nb_filter[0]+nb_filter[1], nb_filter[0])
        #         self.conv1_1 = DoubleConv(nb_filter[1]+nb_filter[2], nb_filter[1])
        #         self.conv2_1 = DoubleConv(nb_filter[2]+nb_filter[3], nb_filter[2])
        #         self.conv3_1 = DoubleConv(nb_filter[3]+nb_filter[4], nb_filter[3])
        self.conv0_1 = DoubleConv2d(nb_filter[0])
        self.conv1_1 = DoubleConv2d(nb_filter[1])
        self.conv2_1 = DoubleConv2d(nb_filter[2])
        self.conv3_1 = DoubleConv2d(nb_filter[3])

        #         self.conv0_2 = DoubleConv(nb_filter[0]*2+nb_filter[1], nb_filter[0])
        #         self.conv1_2 = DoubleConv(nb_filter[1]*2+nb_filter[2], nb_filter[1])
        #         self.conv2_2 = DoubleConv(nb_filter[2]*2+nb_filter[3], nb_filter[2])
        self.conv0_2 = DoubleConv2d(nb_filter[0])
        self.conv1_2 = DoubleConv2d(nb_filter[1])
        self.conv2_2 = DoubleConv2d(nb_filter[2])

        #         self.conv0_3 = DoubleConv(nb_filter[0]*3+nb_filter[1], nb_filter[0])
        #         self.conv1_3 = DoubleConv(nb_filter[1]*3+nb_filter[2], nb_filter[1])

        self.conv0_3 = DoubleConv2d(nb_filter[0])
        self.conv1_3 = DoubleConv2d(nb_filter[1])

        #         self.conv0_4 = DoubleConv(nb_filter[0]*4+nb_filter[1], nb_filter[0])
        self.conv0_4 = DoubleConv2d(nb_filter[0])
        #         self.sigmoid = nn.Sigmoid()

        #         if self.args.deepsupervision:
        #             self.final1 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
        #             self.final2 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
        #             self.final3 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
        #             self.final4 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
        #         else:
        #             self.final = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)

        #         self.conv10 = Conv2D(1,1)
        self.final = Conv2D(1, 1)
        self.sigmoid = Activation('sigmoid')

    def call(self, x):
        #         print("---------show in call func----------")

        #         x0_0 = self.conv0_0(input)
        #         x1_0 = self.conv1_0(self.pool(x0_0))
        #         x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        # -------------------checked--------------------
        #         x00 = self.conv0_0(x)
        #         print("x00.shape=",x00.shape)#         x00 = self.pool(x00)#         print("x00.shape=",x00.shape)
        #         x10 = self.conv1_0(self.pool(x00))
        #         print("x10.shape=",x10.shape)
        #         x01 = self.conv0_1(concatenate([x00,self.up0_1(x10)]))   #   merge6 = concatenate([up_6 , c4])
        #         up = self.up0_1(x10)
        #         print("up.shape",up.shape)
        #         t=concatenate([x00,self.up0_1(x10)])
        #         print("concatenate.shape",t.shape)
        #         print("x01.shape=",x01.shape)
        # --------------------checked------------------------
        x0_0 = self.conv0_0(x)
        if self.print == True:
            print("x0_0.shape=", x0_0.shape)  # print
        p0_0 = self.pool(x0_0)
        if self.dropout == True:
            p0_0 = Dropout(dropoutrate)(p0_0)  # dropout
        x1_0 = self.conv1_0(p0_0)
        #         x1_0 = self.conv1_0(self.pool(x0_0)) #拆成上面的形式因为要dropout  dropout的位置 1.pool后 2.transpose+cat后
        if self.print == True:
            print("x1_0.shape=", x1_0.shape)  # print
        tc0_1 = concatenate([x0_0, self.up0_1(x1_0)])  # tc1_0 表示 t:transpose c：concat 0_1 ：tc到0_1节点
        if self.dropout == True:
            tc0_1 = Dropout(dropoutrate)(tc0_1)  # dropout
        x0_1 = self.conv0_1(tc0_1)
        #         x0_1 = self.conv0_1(concatenate([x0_0,self.up0_1(x1_0)]))
        if self.print == True:
            print("x0_1.shape=", x0_1.shape)  # print
        #         x2_0 = self.conv2_0(self.pool(x1_0))
        #         x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        #         x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        p1_0 = self.pool(x1_0)
        if self.dropout == True:
            p1_0 = Dropout(dropoutrate)(p1_0)  # dropout
        x2_0 = self.conv2_0(p1_0)
        #         x2_0 = self.conv2_0(self.pool(x1_0))
        if self.print == True:
            print("x2_0.shape=", x2_0.shape)  # print
        tc1_1 = concatenate([x1_0, self.up1_1(x2_0)])  # tc1_0 表示 t:transpose c：concat 0_1 ：tc到0_1节点
        if self.dropout == True:
            tc1_1 = Dropout(dropoutrate)(tc1_1)  # dropout
        x1_1 = self.conv1_1(tc1_1)
        #         x1_1 = self.conv1_1(concatenate([x1_0, self.up1_1(x2_0)]))
        if self.print == True:
            print("x1_1.shape=", x1_1.shape)  # print
        tc0_2 = concatenate([x0_0, x0_1, self.up0_2(x1_1)])  # tc1_0 表示 t:transpose c：concat 0_1 ：tc到0_1节点
        if self.dropout == True:
            tc0_2 = Dropout(dropoutrate)(tc0_2)  # dropout
        x0_2 = self.conv0_2(tc0_2)
        #         x0_2 = self.conv0_2(concatenate([x0_0, x0_1, self.up0_2(x1_1)]))
        if self.print == True:
            print("x0_2.shape=", x0_2.shape)  # print
        #         x3_0 = self.conv3_0(self.pool(x2_0))
        #         x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        #         x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        #         x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        p2_0 = self.pool(x2_0)
        if self.dropout == True:
            p2_0 = Dropout(dropoutrate)(p2_0)  # dropout
        x3_0 = self.conv3_0(p2_0)
        #         x3_0 = self.conv3_0(self.pool(x2_0))
        if self.print == True:
            print("x3_0.shape=", x3_0.shape)  # print
        tc2_1 = concatenate([x2_0, self.up2_1(x3_0)])  # tc1_0 表示 t:transpose c：concat 0_1 ：tc到0_1节点
        if self.dropout == True:
            tc2_1 = Dropout(dropoutrate)(tc2_1)  # dropout
        x2_1 = self.conv2_1(tc2_1)
        #         x2_1 = self.conv2_1(concatenate([x2_0, self.up2_1(x3_0)]))
        if self.print == True:
            print("x2_1.shape=", x2_1.shape)  # print
        tc1_2 = concatenate([x1_0, x1_1, self.up1_2(x2_1)])  # tc1_0 表示 t:transpose c：concat 0_1 ：tc到0_1节点
        if self.dropout == True:
            tc1_2 = Dropout(dropoutrate)(tc1_2)  # dropout
        x1_2 = self.conv1_2(tc1_2)
        #         x1_2 = self.conv1_2(concatenate([x1_0, x1_1, self.up1_2(x2_1)]))
        if self.print == True:
            print("x1_2.shape=", x1_2.shape)  # print
        tc0_3 = concatenate([x0_0, x0_1, x0_2, self.up0_3(x1_2)])  # tc1_0 表示 t:transpose c：concat 0_1 ：tc到0_1节点
        if self.dropout == True:
            tc0_3 = Dropout(dropoutrate)(tc0_3)  # dropout
        x0_3 = self.conv0_3(tc0_3)
        #         x0_3 = self.conv0_3(concatenate([x0_0, x0_1, x0_2, self.up0_3(x1_2)]))
        if self.print == True:
            print("x0_3.shape=", x0_3.shape)  # print
        #         x4_0 = self.conv4_0(self.pool(x3_0))
        #         x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        #         x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        #         x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        #         x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        p3_0 = self.pool(x3_0)
        if self.dropout == True:
            p3_0 = Dropout(dropoutrate)(p3_0)  # dropout
        x4_0 = self.conv4_0(p3_0)
        #         x4_0 = self.conv4_0(self.pool(x3_0))
        if self.print == True:
            print("x4_0.shape=", x4_0.shape)  # print
        tc3_1 = concatenate([x3_0, self.up3_1(x4_0)])  # tc1_0 表示 t:transpose c：concat 0_1 ：tc到0_1节点
        if self.dropout == True:
            tc3_1 = Dropout(dropoutrate)(tc3_1)  # dropout
        x3_1 = self.conv3_1(tc3_1)
        #         x3_1 = self.conv3_1(concatenate([x3_0, self.up3_1(x4_0)]))
        if self.print == True:
            print("x3_1.shape=", x3_1.shape)  # print
        tc2_2 = concatenate([x2_0, x2_1, self.up2_2(x3_1)])  # tc1_0 表示 t:transpose c：concat 0_1 ：tc到0_1节点
        if self.dropout == True:
            tc2_2 = Dropout(dropoutrate)(tc2_2)  # dropout
        x2_2 = self.conv2_2(tc2_2)
        #         x2_2 = self.conv2_2(concatenate([x2_0, x2_1, self.up2_2(x3_1)]))
        if self.print == True:
            print("x2_2.shape=", x2_2.shape)  # print
        tc1_3 = concatenate([x1_0, x1_1, x1_2, self.up1_3(x2_2)])  # tc1_0 表示 t:transpose c：concat 0_1 ：tc到0_1节点
        if self.dropout == True:
            tc1_3 = Dropout(dropoutrate)(tc1_3)  # dropout
        x1_3 = self.conv1_3(tc1_3)
        #         x1_3 = self.conv1_3(concatenate([x1_0, x1_1, x1_2, self.up1_3(x2_2)]))
        if self.print == True:
            print("x1_3.shape=", x1_3.shape)  # print
        tc0_4 = concatenate([x0_0, x0_1, x0_2, x0_3, self.up0_4(x1_3)])  # tc1_0 表示 t:transpose c：concat 0_1 ：tc到0_1节点
        if self.dropout == True:
            tc0_4 = Dropout(dropoutrate)(tc0_4)  # dropout
        x0_4 = self.conv0_4(tc0_4)
        #         x0_4 = self.conv0_4(concatenate([x0_0, x0_1, x0_2, x0_3, self.up0_4(x1_3)]))
        if self.print == True:
            print("x0_4.shape=", x0_4.shape)  # print

        #         if self.args.deepsupervision:
        #             output1 = self.final1(x0_1)
        #             output1 = self.sigmoid(output1)
        #             output2 = self.final2(x0_2)
        #             output2 = self.sigmoid(output2)
        #             output3 = self.final3(x0_3)
        #             output3 = self.sigmoid(output3)
        #             output4 = self.final4(x0_4)
        #             output4 = self.sigmoid(output4)
        #             return [output1, output2, output3, output4]

        #         else:
        #             output = self.final(x0_4)
        #             output = self.sigmoid(output)
        #             return output

        # 直接跑全部 不用剪枝了
        output = self.final(x0_4)
        output = self.sigmoid(output)
        return output
# model = NestedUNet()
# # print(model(test).shape)
# callbacks = [
#     #如果accuracy在迭代50次之后还没改变，即停止训练
#     EarlyStopping(patience=50, verbose=1),
#     #学习率不改变的时候就将学习率往下调，此处是将原学习率*0.1,最小的学习率为0.000001
# #     ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.000001, verbose=1),
#     ReduceLROnPlateau(monitor='val_loss',factor=0.1, patience=3, min_lr=0.000001, verbose=1),
#     #keras的模型能够保存为计算机文件，格式为h5，在此处可以设置只保存在训练过程中表现最好的模型
#     #.h5 .hdf5 是一个文件类型的
#     ModelCheckpoint('model-unet-myunetpp.h5', verbose=1, save_best_only=True, save_weights_only=True)
# ]
# model.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy"])

# NO6 Unet(基础版)
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)#批量标准化
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)#激活函数
    return x
def convonce(input_tensor, n_filters, kernel_size=3,strides=1, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), strides=strides,kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)#批量标准化
    x = Activation("relu")(x)
    return x


def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    # 四次最大池化下采样
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)  # 卷积
    p1 = MaxPooling2D((2, 2))(c1)  # 池化
    p1 = Dropout(dropout * 0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # expansive path
    # 四次上采样
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])  # 整个模型的构造
    return model


def get_unet0908v1(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    # 四次最大池化下采样
    cc1 = convonce(input_img, n_filters=n_filters * 2, kernel_size=3, strides=2, batchnorm=batchnorm)  # 卷积
    cc2 = convonce(cc1, n_filters=n_filters * 4, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    cc3 = convonce(cc2, n_filters=n_filters * 8, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    cc4 = convonce(cc3, n_filters=n_filters * 16, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
#     print("cc1,cc2,cc3,cc4.shape:",cc1.shape,cc2.shape,cc3.shape,cc4.shape)
    
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)  # 卷积
    
#     print("c1.shape:",c1.shape)
    p1 = MaxPooling2D((2, 2))(c1)  # 池化
    p1 = Dropout(dropout * 0.5)(p1)
#     print("p1.shape:",p1.shape)
    p1 = concatenate([cc1, p1])
#     print("cc1+p1.shape:",p1.shape)
    
    
    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
#     print("c2.shape:",c2.shape)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
#     print("p2.shape:",p2.shape)
    p2 = concatenate([cc2, p2])
#     print("cc2+p2.shape:",p2.shape)
    
    
    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
#     print("c3.shape:",c3.shape)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
#     print("p3.shape:",p3.shape)
    p3 = concatenate([cc3, p3])
#     print("cc3+p3.shape:",p3.shape)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
#     print("c4.shape:",c4.shape)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)
#     print("p4.shape:",p4.shape)
    p4 = concatenate([cc4, p4])
#     print("cc4+p4.shape:",p4.shape)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)
#     print("c5.shape:",c5.shape)

    # expansive path
    # 四次上采样
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
#     print("u6.shape:",u6.shape)
    c4u6 = concatenate([c4, u6])    
#     print("c4u6.shape:",c4u6.shape)
    #-----------------------------
    c4u6u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c4u6)
#     print("c4u6u7.shape:",c4u6u7.shape)
#     c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(4, 4), padding='same')(c4u6)
    c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c4u6)
    c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c4u6u8)
#     print("c4u6u8.shape:",c4u6u8.shape)
#     c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(8, 8), padding='same')(c4u6)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6u9)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6u9)
#     print("c4u6u9.shape:",c4u6u9.shape)
    #-----------------------------
    u6 = Dropout(dropout)(c4u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
#     print("c6.shape:",c6.shape)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
#     print("u7.shape:",u7.shape)
#     c3u7 = concatenate([c3, u7])
    c3u7 = concatenate([c3, u7,c4u6u7])
#     print("c3u7.shape:",c3u7.shape)
    #-----------------------------
    c3u7u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c3u7)
#     print("c3u7u8.shape:",c3u7u8.shape)
#     c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(4, 4), padding='same')(c3u7)
    c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c3u7)
    c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c3u7u9)
#     print("c3u7u9.shape:",c3u7u9.shape)
    #-----------------------------
    u7 = Dropout(dropout)(c3u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
#     print("c7.shape:",c7.shape)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
#     print("u8.shape:",u8.shape)
#     c2u8 = concatenate([c2, u8])
    c2u8 = concatenate([c2, u8,c4u6u8,c3u7u8])
#     print("c2u8.shape:",c2u8.shape)
    #-----------------------------
    c2u8u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c2u8)
#     print("c2u8u9.shape:",c2u8u9.shape)
    #-----------------------------
    u8 = Dropout(dropout)(c2u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
#     print("c8.shape:",c8.shape)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
#     print("u9.shape:",u9.shape)
#     c1u9 = concatenate([c1, u9], axis=3)
    c1u9 = concatenate([c1, u9,c4u6u9,c3u7u9,c2u8u9], axis=3)
#     print("c1u9.shape:",c1u9.shape)
    u9 = Dropout(dropout)(c1u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
#     print("c9.shape:",c9.shape)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])  # 整个模型的构造
    return model
def get_unet0908v2(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    #  V2 先比V1 从下往上跳接时 只用 上采样的 不用到merge后的
    # contracting path
    # 四次最大池化下采样
    cc1 = convonce(input_img, n_filters=n_filters * 2, kernel_size=3, strides=2, batchnorm=batchnorm)  # 卷积
    cc2 = convonce(cc1, n_filters=n_filters * 4, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    cc3 = convonce(cc2, n_filters=n_filters * 8, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    cc4 = convonce(cc3, n_filters=n_filters * 16, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
#     print("cc1,cc2,cc3,cc4.shape:",cc1.shape,cc2.shape,cc3.shape,cc4.shape)
    
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)  # 卷积
    
#     print("c1.shape:",c1.shape)
    p1 = MaxPooling2D((2, 2))(c1)  # 池化
    p1 = Dropout(dropout * 0.5)(p1)
#     print("p1.shape:",p1.shape)
    p1 = concatenate([cc1, p1])
#     print("cc1+p1.shape:",p1.shape)
    
    
    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
#     print("c2.shape:",c2.shape)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
#     print("p2.shape:",p2.shape)
    p2 = concatenate([cc2, p2])
#     print("cc2+p2.shape:",p2.shape)
    
    
    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
#     print("c3.shape:",c3.shape)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
#     print("p3.shape:",p3.shape)
    p3 = concatenate([cc3, p3])
#     print("cc3+p3.shape:",p3.shape)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
#     print("c4.shape:",c4.shape)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)
#     print("p4.shape:",p4.shape)
    p4 = concatenate([cc4, p4])
#     print("cc4+p4.shape:",p4.shape)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)
#     print("c5.shape:",c5.shape)

    # expansive path
    # 四次上采样
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
#     print("u6.shape:",u6.shape)
    c4u6 = concatenate([c4, u6])    
#     print("c4u6.shape:",c4u6.shape)
    #-----------------------------
    u6u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(u6)
#     print("u6u7.shape:",u6u7.shape)
#     c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(4, 4), padding='same')(c4u6)
    u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(u6)
    u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(u6u8)
#     print("u6u8.shape:",u6u8.shape)
#     c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(8, 8), padding='same')(c4u6)
    u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(u6)
    u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(u6u9)
    u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(u6u9)
#     print("u6u9.shape:",u6u9.shape)
    #-----------------------------
    c4u6 = Dropout(dropout)(c4u6)
    c6 = conv2d_block(c4u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
#     print("c6.shape:",c6.shape)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
#     print("u7.shape:",u7.shape)
#     c3u7 = concatenate([c3, u7])
    c3u7 = concatenate([c3, u7,u6u7])
#     print("c3u7.shape:",c3u7.shape)
    #-----------------------------
    u7u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(u7)
#     print("u7u8.shape:",u7u8.shape)
#     c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(4, 4), padding='same')(c3u7)
    u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(u7)
    u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(u7u9)
#     print("u7u9.shape:",u7u9.shape)
    #-----------------------------
    c3u7 = Dropout(dropout)(c3u7)
    c7 = conv2d_block(c3u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
#     print("c7.shape:",c7.shape)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
#     print("u8.shape:",u8.shape)
#     c2u8 = concatenate([c2, u8])
    c2u8 = concatenate([c2, u8,u6u8,u7u8])
#     print("c2u8.shape:",c2u8.shape)
    #-----------------------------
    u8u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(u8)
#     print("u8u9.shape:",u8u9.shape)
    #-----------------------------
    c2u8 = Dropout(dropout)(c2u8)
    c8 = conv2d_block(c2u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
#     print("c8.shape:",c8.shape)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
#     print("u9.shape:",u9.shape)
#     c1u9 = concatenate([c1, u9], axis=3)
    c1u9 = concatenate([c1, u9,u6u9,u7u9,u8u9], axis=3)
#     print("c1u9.shape:",c1u9.shape)
    c1u9 = Dropout(dropout)(c1u9)
    c9 = conv2d_block(c1u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
#     print("c9.shape:",c9.shape)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])  # 整个模型的构造
    return model
def get_unet_att_convt0910(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    # 四次最大池化下采样
    cc1 = convonce(input_img, n_filters=n_filters * 2, kernel_size=3, strides=2, batchnorm=batchnorm)  # 卷积
    cc2 = convonce(cc1, n_filters=n_filters * 4, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    cc3 = convonce(cc2, n_filters=n_filters * 8, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    cc4 = convonce(cc3, n_filters=n_filters * 16, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
#     print("cc1,cc2,cc3,cc4.shape:",cc1.shape,cc2.shape,cc3.shape,cc4.shape)
    
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)  # 卷积
    
#     print("c1.shape:",c1.shape)
    p1 = MaxPooling2D((2, 2))(c1)  # 池化
    p1 = Dropout(dropout * 0.5)(p1)
#     print("p1.shape:",p1.shape)
    p1 = concatenate([cc1, p1])
#     print("cc1+p1.shape:",p1.shape)
    
    
    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
#     print("c2.shape:",c2.shape)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
#     print("p2.shape:",p2.shape)
    p2 = concatenate([cc2, p2])
#     print("cc2+p2.shape:",p2.shape)
    
    
    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
#     print("c3.shape:",c3.shape)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
#     print("p3.shape:",p3.shape)
    p3 = concatenate([cc3, p3])
#     print("cc3+p3.shape:",p3.shape)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
#     print("c4.shape:",c4.shape)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)
#     print("p4.shape:",p4.shape)
    p4 = concatenate([cc4, p4])
#     print("cc4+p4.shape:",p4.shape)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)
#     print("c5.shape:",c5.shape)

    # expansive path
    # 四次上采样
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
#     print("u6.shape:",u6.shape)
    #---------------------att blk----
    att4 = Attention_block(inputg=u6, inputx=c4, F_int=n_filters * 4)
#     print("att4=u6+c4:",att4.shape)
    #---------------------------------
    c4u6 = concatenate([att4, u6])    
#     print("attc4-c4u6.shape:",c4u6.shape)
    #-----------------------------
    c4u6u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c4u6)
#     print("c4u6u7.shape:",c4u6u7.shape)
#     c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(4, 4), padding='same')(c4u6)
    c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c4u6)
    c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c4u6u8)
#     print("c4u6u8.shape:",c4u6u8.shape)
#     c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(8, 8), padding='same')(c4u6)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6u9)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6u9)
#     print("c4u6u9.shape:",c4u6u9.shape)
    #-----------------------------
    u6 = Dropout(dropout)(c4u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
#     print("c6.shape:",c6.shape)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
#     print("u7.shape:",u7.shape)
    #---------------------att blk----
    att3 = Attention_block(inputg=u7, inputx=c3, F_int=n_filters * 2)
#     print("att3=u7+c3:",att3.shape)
    #---------------------------------
#     c3u7 = concatenate([c3, u7])
    c3u7 = concatenate([att3, u7,c4u6u7])
#     print("attc3-c3u7.shape:",c3u7.shape)
    #-----------------------------
    c3u7u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c3u7)
#     print("c3u7u8.shape:",c3u7u8.shape)
#     c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(4, 4), padding='same')(c3u7)
    c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c3u7)
    c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c3u7u9)
#     print("c3u7u9.shape:",c3u7u9.shape)
    #-----------------------------
    u7 = Dropout(dropout)(c3u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
#     print("c7.shape:",c7.shape)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
#     print("u8.shape:",u8.shape) 
    #---------------------att blk----
    att2 = Attention_block(inputg=u8, inputx=c2, F_int=n_filters * 1)
#     print("att2=u8+c2:",att2.shape)
    #---------------------------------
    #-----------------------------
#     c2u8 = concatenate([c2, u8])
    c2u8 = concatenate([att2, u8,c4u6u8,c3u7u8])
#     print("attc2-c2u8.shape:",c2u8.shape)
    #-----------------------------
    c2u8u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c2u8)
#     print("c2u8u9.shape:",c2u8u9.shape)
    #-----------------------------
    u8 = Dropout(dropout)(c2u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
#     print("c8.shape:",c8.shape)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
#     print("u9.shape:",u9.shape)
    #---------------------att blk----
    att1 = Attention_block(inputg=u9, inputx=c1, F_int=n_filters /2)
#     print("att1=u9+c1:",att2.shape)
    #---------------------------------
#     c1u9 = concatenate([c1, u9], axis=3)
    c1u9 = concatenate([att1, u9,c4u6u9,c3u7u9,c2u8u9], axis=3)
#     print("attc2-c2u8.shape:",c1u9.shape)
    u9 = Dropout(dropout)(c1u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
#     print("c9.shape:",c9.shape)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])  # 整个模型的构造
    return model
def att_unet_with_cc1(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    #不加dense模块 就平常的att+前面的cc1
    # contracting path
    # 四次最大池化下采样
    cc1 = convonce(input_img, n_filters=n_filters * 2, kernel_size=3, strides=2, batchnorm=batchnorm)  # 卷积
    cc2 = convonce(cc1, n_filters=n_filters * 4, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    cc3 = convonce(cc2, n_filters=n_filters * 8, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    cc4 = convonce(cc3, n_filters=n_filters * 16, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
#     print("cc1,cc2,cc3,cc4.shape:",cc1.shape,cc2.shape,cc3.shape,cc4.shape)
    
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)  # 卷积
    
#     print("c1.shape:",c1.shape)
    p1 = MaxPooling2D((2, 2))(c1)  # 池化
    p1 = Dropout(dropout * 0.5)(p1)
#     print("p1.shape:",p1.shape)
    p1 = concatenate([cc1, p1])
#     print("cc1+p1.shape:",p1.shape)
    
    
    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
#     print("c2.shape:",c2.shape)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
#     print("p2.shape:",p2.shape)
    p2 = concatenate([cc2, p2])
#     print("cc2+p2.shape:",p2.shape)
    
    
    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
#     print("c3.shape:",c3.shape)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
#     print("p3.shape:",p3.shape)
    p3 = concatenate([cc3, p3])
#     print("cc3+p3.shape:",p3.shape)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
#     print("c4.shape:",c4.shape)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)
#     print("p4.shape:",p4.shape)
    p4 = concatenate([cc4, p4])
#     print("cc4+p4.shape:",p4.shape)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)
#     print("c5.shape:",c5.shape)

    # expansive path
    # 四次上采样
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
#     print("u6.shape:",u6.shape)
    #---------------------att blk----
    att4 = Attention_block(inputg=u6, inputx=c4, F_int=n_filters * 4)
#     print("att4=u6+c4:",att4.shape)
    #---------------------------------
    c4u6 = concatenate([att4, u6])    
#     print("attc4-c4u6.shape:",c4u6.shape)
    #-----------------------------
    #-----------------------------
    u6 = Dropout(dropout)(c4u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
#     print("c6.shape:",c6.shape)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
#     print("u7.shape:",u7.shape)
    #---------------------att blk----
    att3 = Attention_block(inputg=u7, inputx=c3, F_int=n_filters * 2)
#     print("att3=u7+c3:",att3.shape)
    #---------------------------------
#     c3u7 = concatenate([c3, u7])
    c3u7 = concatenate([att3, u7])
#     print("attc3-c3u7.shape:",c3u7.shape)
    #-----------------------------
    #-----------------------------
    u7 = Dropout(dropout)(c3u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
#     print("c7.shape:",c7.shape)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
#     print("u8.shape:",u8.shape) 
    #---------------------att blk----
    att2 = Attention_block(inputg=u8, inputx=c2, F_int=n_filters * 1)
#     print("att2=u8+c2:",att2.shape)
    #---------------------------------
    #-----------------------------
    c2u8 = concatenate([c2, u8])
#     print("attc2-c2u8.shape:",c2u8.shape)
    #-----------------------------
    #-----------------------------
    u8 = Dropout(dropout)(c2u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
#     print("c8.shape:",c8.shape)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
#     print("u9.shape:",u9.shape)
    #---------------------att blk----
    att1 = Attention_block(inputg=u9, inputx=c1, F_int=n_filters /2)
#     print("att1=u9+c1:",att1.shape)
    #---------------------------------
#     c1u9 = concatenate([c1, u9], axis=3)
    c1u9 = concatenate([att1, u9], axis=3)
#     print("attc1-c1u9.shape:",c1u9.shape)
    u9 = Dropout(dropout)(c1u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
#     print("c9.shape:",c9.shape)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])  # 整个模型的构造
    return model
# 弄一下Unet模块
def get_unet_withcc1_only0908(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    # 四次最大池化下采样
    cc1 = convonce(input_img, n_filters=n_filters * 2, kernel_size=3, strides=2, batchnorm=batchnorm)  # 卷积
    cc2 = convonce(cc1, n_filters=n_filters * 4, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    cc3 = convonce(cc2, n_filters=n_filters * 8, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    cc4 = convonce(cc3, n_filters=n_filters * 16, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
#     print("cc1,cc2,cc3,cc4.shape:",cc1.shape,cc2.shape,cc3.shape,cc4.shape)
    
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)  # 卷积
    
#     print("c1.shape:",c1.shape)
    p1 = MaxPooling2D((2, 2))(c1)  # 池化
    p1 = Dropout(dropout * 0.5)(p1)
#     print("p1.shape:",p1.shape)
    p1 = concatenate([cc1, p1])
#     print("cc1+p1.shape:",p1.shape)
    
    
    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
#     print("c2.shape:",c2.shape)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
#     print("p2.shape:",p2.shape)
    p2 = concatenate([cc2, p2])
#     print("cc2+p2.shape:",p2.shape)
    
    
    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
#     print("c3.shape:",c3.shape)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
#     print("p3.shape:",p3.shape)
    p3 = concatenate([cc3, p3])
#     print("cc3+p3.shape:",p3.shape)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
#     print("c4.shape:",c4.shape)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)
#     print("p4.shape:",p4.shape)
    p4 = concatenate([cc4, p4])
#     print("cc4+p4.shape:",p4.shape)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)
#     print("c5.shape:",c5.shape)

    # expansive path
    # 四次上采样
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
#     print("u6.shape:",u6.shape)
    c4u6 = concatenate([c4, u6])    
#     print("c4u6.shape:",c4u6.shape)
    #-----------------------------
#     print("c4u6u9.shape:",c4u6u9.shape)
    #-----------------------------
    u6 = Dropout(dropout)(c4u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
#     print("c6.shape:",c6.shape)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
#     print("u7.shape:",u7.shape)
#     c3u7 = concatenate([c3, u7])
    c3u7 = concatenate([c3, u7])
#     print("c3u7.shape:",c3u7.shape)
    #-----------------------------
#     print("c3u7u9.shape:",c3u7u9.shape)
    #-----------------------------
    u7 = Dropout(dropout)(c3u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
#     print("c7.shape:",c7.shape)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
#     print("u8.shape:",u8.shape)
#     c2u8 = concatenate([c2, u8])
    c2u8 = concatenate([c2, u8])
#     print("c2u8.shape:",c2u8.shape)
    #-----------------------------
  
    #-----------------------------
    u8 = Dropout(dropout)(c2u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
#     print("c8.shape:",c8.shape)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
#     print("u9.shape:",u9.shape)
#     c1u9 = concatenate([c1, u9], axis=3)
    c1u9 = concatenate([c1, u9], axis=3)
#     print("c1u9.shape:",c1u9.shape)
    u9 = Dropout(dropout)(c1u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
#     print("c9.shape:",c9.shape)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])  # 整个模型的构造
    return model
def get_unet_withdenseonly_0908(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)  # 卷积
    
#     print("c1.shape:",c1.shape)
    p1 = MaxPooling2D((2, 2))(c1)  # 池化
    p1 = Dropout(dropout * 0.5)(p1)
#     print("p1.shape:",p1.shape)
    
    
    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
#     print("c2.shape:",c2.shape)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
#     print("p2.shape:",p2.shape)
    
    
    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
#     print("c3.shape:",c3.shape)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
#     print("p3.shape:",p3.shape)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
#     print("c4.shape:",c4.shape)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)
#     print("p4.shape:",p4.shape)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)
#     print("c5.shape:",c5.shape)

    # expansive path
    # 四次上采样
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
#     print("u6.shape:",u6.shape)
    c4u6 = concatenate([c4, u6])    
#     print("c4u6.shape:",c4u6.shape)
    #-----------------------------
    c4u6u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c4u6)
#     print("c4u6u7.shape:",c4u6u7.shape)
#     c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(4, 4), padding='same')(c4u6)
    c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c4u6)
    c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c4u6u8)
#     print("c4u6u8.shape:",c4u6u8.shape)
#     c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(8, 8), padding='same')(c4u6)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6u9)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6u9)
#     print("c4u6u9.shape:",c4u6u9.shape)
    #-----------------------------
    u6 = Dropout(dropout)(c4u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
#     print("c6.shape:",c6.shape)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
#     print("u7.shape:",u7.shape)
#     c3u7 = concatenate([c3, u7])
    c3u7 = concatenate([c3, u7,c4u6u7])
#     print("c3u7.shape:",c3u7.shape)
    #-----------------------------
    c3u7u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c3u7)
#     print("c3u7u8.shape:",c3u7u8.shape)
#     c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(4, 4), padding='same')(c3u7)
    c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c3u7)
    c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c3u7u9)
#     print("c3u7u9.shape:",c3u7u9.shape)
    #-----------------------------
    u7 = Dropout(dropout)(c3u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
#     print("c7.shape:",c7.shape)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
#     print("u8.shape:",u8.shape)
#     c2u8 = concatenate([c2, u8])
    c2u8 = concatenate([c2, u8,c4u6u8,c3u7u8])
#     print("c2u8.shape:",c2u8.shape)
    #-----------------------------
    c2u8u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c2u8)
#     print("c2u8u9.shape:",c2u8u9.shape)
    #-----------------------------
    u8 = Dropout(dropout)(c2u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
#     print("c8.shape:",c8.shape)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
#     print("u9.shape:",u9.shape)
#     c1u9 = concatenate([c1, u9], axis=3)
    c1u9 = concatenate([c1, u9,c4u6u9,c3u7u9,c2u8u9], axis=3)
#     print("c1u9.shape:",c1u9.shape)
    u9 = Dropout(dropout)(c1u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
#     print("c9.shape:",c9.shape)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])  # 整个模型的构造
    return model
# 卷积所使用的深度
base1 = 16  # 64我的电脑肯定跑不动
# [64,128,256,512,1024]    [16,32,64,128,256]
filtercount = [base1 * 1, base1 * 2, base1 * 4, base1 * 8, base1 * 16]
# 搞一下dropout
dropoutrate = 0.25


class Unet(Model):
    #     def __init__(self):
    def __init__(self):
        super(Unet, self).__init__()
        self.dropout = True

        self.conv1 = DoubleConv2d(filtercount[0])
        self.pool1 = MaxPool2D(pool_size=(2, 2), strides=2)  # 池化层
        self.conv2 = DoubleConv2d(filtercount[1])
        self.pool2 = MaxPool2D(2)
        self.conv3 = DoubleConv2d(filtercount[2])
        self.pool3 = MaxPool2D(2)
        self.conv4 = DoubleConv2d(filtercount[3])
        self.pool4 = MaxPool2D(2)
        self.conv5 = DoubleConv2d(filtercount[4])
        # https://blog.csdn.net/coolsunxu/article/details/106317157
        self.up6 = Conv2DTranspose(filters=filtercount[3], kernel_size=2, strides=2)
        self.conv6 = DoubleConv2d(filtercount[3])
        self.up7 = Conv2DTranspose(filters=filtercount[2], kernel_size=2, strides=2)
        self.conv7 = DoubleConv2d(filtercount[2])
        self.up8 = Conv2DTranspose(filters=filtercount[1], kernel_size=2, strides=2)
        self.conv8 = DoubleConv2d(filtercount[1])
        self.up9 = Conv2DTranspose(filters=filtercount[0], kernel_size=2, strides=2)
        self.conv9 = DoubleConv2d(filtercount[0])
        # Conv2D(1, (1, 1), activation='sigmoid') (c9)
        self.conv10 = Conv2D(1, 1)

    def call(self, x):
        # down
        #         print("x.shape",x.shape)
        c1 = self.conv1(x)
        #         print("c1.shape",c1.shape)
        p1 = self.pool1(c1)
        #         print("p1.shape",p1.shape)
        if self.dropout == True:
            p1 = Dropout(dropoutrate)(p1)  # 这里取 d1的话 就不好改啦

        c2 = self.conv2(p1)
        #         print("c2.shape",c2.shape)
        p2 = self.pool2(c2)
        #         print("p2.shape",p2.shape)

        if self.dropout == True:
            p2 = Dropout(dropoutrate)(p2)  # dropout

        c3 = self.conv3(p2)
        #         print("c3.shape",c3.shape)
        p3 = self.pool3(c3)
        #         print("p3.shape",p3.shape)
        if self.dropout == True:
            p3 = Dropout(dropoutrate)(p3)  # dropout

        c4 = self.conv4(p3)
        #         print("c4.shape",c4.shape)
        p4 = self.pool4(c4)
        #         print("p4.shape",p4.shape)
        if self.dropout == True:
            p4 = Dropout(dropoutrate)(p4)  # dropout
        c5 = self.conv5(p4)
        #         print("c5.shape",c5.shape)

        # up
        up_6 = self.up6(c5)
        #         print("up_6.shape",up_6.shape)

        merge6 = concatenate([up_6, c4])
        #         print("merge6.shape",merge6.shape)
        if self.dropout == True:
            merge6 = Dropout(dropoutrate)(merge6)  # dropout
        c6 = self.conv6(merge6)
        #         print("c6.shape",c6.shape)

        up_7 = self.up7(c6)
        #         print("up_7.shape",up_7.shape)
        merge7 = concatenate([up_7, c3])
        #         print("merge7.shape",merge7.shape)
        if self.dropout == True:
            merge7 = Dropout(dropoutrate)(merge7)  # dropout

        c7 = self.conv7(merge7)
        #         print("c7.shape",c7.shape)

        up_8 = self.up8(c7)
        #         print("up_8.shape",up_8.shape)
        merge8 = concatenate([up_8, c2])
        #         print("merge8.shape",merge6.shape)
        if self.dropout == True:
            merge8 = Dropout(dropoutrate)(merge8)  # dropout
        c8 = self.conv8(merge8)
        #         print("c8.shape",c8.shape)

        up_9 = self.up9(c8)
        #         print("up_9.shape",up_9.shape)
        merge9 = concatenate([up_9, c1])
        #         print("merge6.shape",merge9.shape)
        if self.dropout == True:
            merge9 = Dropout(dropoutrate)(merge9)  # dropout
        c9 = self.conv9(merge9)
        #         print("c9.shape",c9.shape)

        c10 = self.conv10(c9)
        #         print("c10.shape",c10.shape)

        out = Activation("sigmoid")(c10)

        return out
# NO7. BiONet
class BiONet(object):

    def __init__(self,
                 input_shape,
                 num_classes=1,
                 iterations=2,
                 multiplier=1.0,
                 num_layers=4,
                 integrate=False,
                 nfilters=32):

        activation = 'relu'
        kernel_initializer = 'he_normal'
        padding = 'same'
        kernel_size = (3, 3)
        self.num_layers = num_layers
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.iterations = iterations
        self.multiplier = multiplier
        self.integrate = integrate
        self.nfilters = nfilters

        self.filters_list = [int(nfilters * (2 ** i) * self.multiplier) for i in range(self.num_layers + 1)]
        self.bachnorm_momentum = 0.01

        self.conv_args = {
            'kernel_size': kernel_size,
            'activation': activation,
            'padding': padding,
            'kernel_initializer': kernel_initializer
        }

        self.convT_args = {
            'kernel_size': kernel_size,
            'activation': activation,
            'strides': (2, 2),
            'padding': padding
        }

        self.maxpool_args = {
            'pool_size': (2, 2),
            'strides': (2, 2),
            'padding': 'valid',
        }

    # define reusable layers
    def define_layers(self):

        # reuse feature collections
        conv_layers = []
        deconv_layers = []
        mid_layers = []

        mid_layers.append(Conv2D(self.filters_list[self.num_layers], **self.conv_args))
        mid_layers.append(Conv2D(self.filters_list[self.num_layers], **self.conv_args))
        mid_layers.append(Conv2DTranspose(self.filters_list[self.num_layers], **self.convT_args))

        for l in range(self.num_layers):
            conv_layers.append(Conv2D(self.filters_list[l], **self.conv_args))
            conv_layers.append(Conv2D(self.filters_list[l], **self.conv_args))
            conv_layers.append(Conv2D(self.filters_list[l + 1], **self.conv_args))

        for l in range(self.num_layers):
            deconv_layers.append(Conv2D(self.filters_list[self.num_layers - 1 - l], **self.conv_args))
            deconv_layers.append(Conv2D(self.filters_list[self.num_layers - 1 - l], **self.conv_args))
            deconv_layers.append(Conv2DTranspose(self.filters_list[self.num_layers - 1 - l], **self.convT_args))

        return conv_layers, deconv_layers, mid_layers

    # define BiO-Net graph, reusable layers will be passed in
    def define_graph(self, conv_layers, mid_layers, deconv_layers):

        inputs = Input(self.input_shape)

        # the first stage block is not reusable
        x = self.conv_block(inputs, self.bachnorm_momentum, int(self.nfilters * self.multiplier))
        shortcut = self.conv_block(x, self.bachnorm_momentum, int(self.nfilters * self.multiplier))
        x = self.conv_block(shortcut, self.bachnorm_momentum, int(self.nfilters * self.multiplier))
        x_in = MaxPooling2D(**self.maxpool_args)(x)

        back_layers = []
        collection = []

        for it in range(self.iterations):

            # down layers to carry forward skip connections
            down_layers = []

            # encoding stage
            for l in range(self.num_layers):
                if l == 0:
                    x = x_in

                if len(back_layers) != 0:
                    x = Concatenate()([x, back_layers[-1 - l]])  # backward skip connection
                else:
                    x = Concatenate()([x, x])  # self concatenation if there is no backward connections

                x = self.conv_block(x, self.bachnorm_momentum, conv=conv_layers[3 * l])
                x = self.conv_block(x, self.bachnorm_momentum, conv=conv_layers[3 * l + 1])
                down_layers.append(x)
                x = self.conv_block(x, self.bachnorm_momentum, conv=conv_layers[3 * l + 2])
                x = MaxPooling2D(**self.maxpool_args)(x)

            # back layers to carry backward skip connections, refresh in each inference iteration
            back_layers = []

            # middle stage
            x = self.conv_block(x, self.bachnorm_momentum, conv=mid_layers[0])
            x = self.conv_block(x, self.bachnorm_momentum, conv=mid_layers[1])
            x = self.conv_block(x, self.bachnorm_momentum, conv=mid_layers[2])

            # decoding stage
            for l in range(self.num_layers):
                x = concatenate([x, down_layers[-1 - l]])  # forward skip connection
                x = self.conv_block(x, self.bachnorm_momentum, conv=deconv_layers[3 * l])
                x = self.conv_block(x, self.bachnorm_momentum, conv=deconv_layers[3 * l + 1])
                back_layers.append(x)
                x = self.conv_block(x, self.bachnorm_momentum, conv=deconv_layers[3 * l + 2])

            # integrate decoded features
            if self.integrate:
                collection.append(x)

        # 'integrate' leads to potential better results
        if self.integrate:
            x = concatenate(collection)

        # the last stage block is not reusable
        x = self.conv_block(x, self.bachnorm_momentum, int(self.nfilters * self.multiplier))
        x = self.conv_block(x, self.bachnorm_momentum, int(self.nfilters * self.multiplier))

        outputs = Conv2D(self.num_classes, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid', padding='valid')(x)

        model = Model(inputs=[inputs], outputs=[outputs])

        return model

    def build(self):
        conv_layers, deconv_layers, mid_layers = self.define_layers()
        model = self.define_graph(conv_layers, mid_layers, deconv_layers)
        return model

    def conv_block(self, x, bachnorm_momentum, filters=None, conv=None):
        if conv is not None:
            x = conv(x)
        else:
            x = Conv2D(filters, **self.conv_args)(x)
        x = BatchNormalization(momentum=bachnorm_momentum)(x)  # BNs are NOT reusable
        return x


# bio-net的评价指标 in training
def iou(y_true, y_pred, threshold=0.5):
    y_pred = threshold_binarize(y_pred, threshold)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection)


def dice_coef(y_true, y_pred, threshold=0.5):
    y_pred = threshold_binarize(y_pred, threshold)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (
            K.sum(y_true_f) + K.sum(y_pred_f))


def threshold_binarize(x, threshold=0.5):
    ge = tf.greater_equal(x, tf.constant(threshold))
    y = tf.where(ge, x=tf.ones_like(x), y=tf.zeros_like(x))
    return y


# create model
# input_img = Input((512, 512, 3),name="imgtest")
x = np.random.randn(512, 512, 3)
input_shape = x.shape
model = BiONet(
    input_shape,
    num_classes=1,  # 1
    num_layers=4,
    iterations=2,  # 1 2 3
    multiplier=1.0,  # 1.0
    integrate=True,  # True
    nfilters=16    # 32
).build()

# # model.compile(
# #   optimizer=Adam(lr=args.lr,decay=args.lr_decay), #0.01     3e-5
# #   loss = 'binary_crossentropy',
# #   metrics=[iou, dice_coef]
# # )
# model.summary(line_length=120, positions=[0.40, 0.60, 0.7, 1.])
# x = np.random.randn(1, 512, 512, 3)
# out = model(x)
# print(out.shape)

# resunet
def res_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)#批量标准化
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)#激活函数
    # 用CONV 1x1 对 inputtensor 扩维后才能和 x相加
    input_tensor = Conv2D(filters=n_filters, kernel_size=(1, 1), kernel_initializer="he_normal",
               padding="same")(x)
#     print("x and input:",x.shape,input_tensor.shape)
    res = input_tensor + x
    return res
def resunet0912(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    # 四次最大池化下采样
    c1 = res_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)  # 卷积
    p1 = MaxPooling2D((2, 2))(c1)  # 池化
    p1 = Dropout(dropout * 0.5)(p1)

    c2 = res_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = res_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = res_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = res_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # expansive path
    # 四次上采样
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = res_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = res_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = res_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = res_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])  # 整个模型的构造
    return model

def resDunet0912(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    # 四次最大池化下采样
    cc1 = convonce(input_img, n_filters=n_filters * 2, kernel_size=3, strides=2, batchnorm=batchnorm)  # 卷积
    cc2 = convonce(cc1, n_filters=n_filters * 4, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    cc3 = convonce(cc2, n_filters=n_filters * 8, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    cc4 = convonce(cc3, n_filters=n_filters * 16, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
#     print("cc1,cc2,cc3,cc4.shape:",cc1.shape,cc2.shape,cc3.shape,cc4.shape)
    
    c1 = res_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)  # 卷积
    
#     print("c1.shape:",c1.shape)
    p1 = MaxPooling2D((2, 2))(c1)  # 池化
    p1 = Dropout(dropout * 0.5)(p1)
#     print("p1.shape:",p1.shape)
    p1 = concatenate([cc1, p1])
#     print("cc1+p1.shape:",p1.shape)
    
    
    c2 = res_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
#     print("c2.shape:",c2.shape)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
#     print("p2.shape:",p2.shape)
    p2 = concatenate([cc2, p2])
#     print("cc2+p2.shape:",p2.shape)
    
    
    c3 = res_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
#     print("c3.shape:",c3.shape)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
#     print("p3.shape:",p3.shape)
    p3 = concatenate([cc3, p3])
#     print("cc3+p3.shape:",p3.shape)

    c4 = res_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
#     print("c4.shape:",c4.shape)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)
#     print("p4.shape:",p4.shape)
    p4 = concatenate([cc4, p4])
#     print("cc4+p4.shape:",p4.shape)

    c5 = res_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)
#     print("c5.shape:",c5.shape)

    # expansive path
    # 四次上采样
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
#     print("u6.shape:",u6.shape)
    c4u6 = concatenate([c4, u6])    
#     print("c4u6.shape:",c4u6.shape)
    #-----------------------------
    c4u6u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c4u6)
#     print("c4u6u7.shape:",c4u6u7.shape)
#     c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(4, 4), padding='same')(c4u6)
    c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c4u6)
    c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c4u6u8)
#     print("c4u6u8.shape:",c4u6u8.shape)
#     c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(8, 8), padding='same')(c4u6)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6u9)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6u9)
#     print("c4u6u9.shape:",c4u6u9.shape)
    #-----------------------------
    u6 = Dropout(dropout)(c4u6)
    c6 = res_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
#     print("c6.shape:",c6.shape)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
#     print("u7.shape:",u7.shape)
#     c3u7 = concatenate([c3, u7])
    c3u7 = concatenate([c3, u7,c4u6u7])
#     print("c3u7.shape:",c3u7.shape)
    #-----------------------------
    c3u7u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c3u7)
#     print("c3u7u8.shape:",c3u7u8.shape)
#     c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(4, 4), padding='same')(c3u7)
    c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c3u7)
    c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c3u7u9)
#     print("c3u7u9.shape:",c3u7u9.shape)
    #-----------------------------
    u7 = Dropout(dropout)(c3u7)
    c7 = res_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
#     print("c7.shape:",c7.shape)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
#     print("u8.shape:",u8.shape)
#     c2u8 = concatenate([c2, u8])
    c2u8 = concatenate([c2, u8,c4u6u8,c3u7u8])
#     print("c2u8.shape:",c2u8.shape)
    #-----------------------------
    c2u8u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c2u8)
#     print("c2u8u9.shape:",c2u8u9.shape)
    #-----------------------------
    u8 = Dropout(dropout)(c2u8)
    c8 = res_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
#     print("c8.shape:",c8.shape)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
#     print("u9.shape:",u9.shape)
#     c1u9 = concatenate([c1, u9], axis=3)
    c1u9 = concatenate([c1, u9,c4u6u9,c3u7u9,c2u8u9], axis=3)
#     print("c1u9.shape:",c1u9.shape)
    u9 = Dropout(dropout)(c1u9)
    c9 = res_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
#     print("c9.shape:",c9.shape)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])  # 整个模型的构造
    return model

def resDunet_cc1_nodense_0912(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    # 四次最大池化下采样
    cc1 = convonce(input_img, n_filters=n_filters * 2, kernel_size=3, strides=2, batchnorm=batchnorm)  # 卷积
    cc2 = convonce(cc1, n_filters=n_filters * 4, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    cc3 = convonce(cc2, n_filters=n_filters * 8, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    cc4 = convonce(cc3, n_filters=n_filters * 16, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
#     print("cc1,cc2,cc3,cc4.shape:",cc1.shape,cc2.shape,cc3.shape,cc4.shape)
    
    c1 = res_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)  # 卷积
    
#     print("c1.shape:",c1.shape)
    p1 = MaxPooling2D((2, 2))(c1)  # 池化
    p1 = Dropout(dropout * 0.5)(p1)
#     print("p1.shape:",p1.shape)
    p1 = concatenate([cc1, p1])
#     print("cc1+p1.shape:",p1.shape)
    
    
    c2 = res_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
#     print("c2.shape:",c2.shape)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
#     print("p2.shape:",p2.shape)
    p2 = concatenate([cc2, p2])
#     print("cc2+p2.shape:",p2.shape)
    
    
    c3 = res_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
#     print("c3.shape:",c3.shape)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
#     print("p3.shape:",p3.shape)
    p3 = concatenate([cc3, p3])
#     print("cc3+p3.shape:",p3.shape)

    c4 = res_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
#     print("c4.shape:",c4.shape)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)
#     print("p4.shape:",p4.shape)
    p4 = concatenate([cc4, p4])
#     print("cc4+p4.shape:",p4.shape)

    c5 = res_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)
#     print("c5.shape:",c5.shape)

    # expansive path
    # 四次上采样
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
#     print("u6.shape:",u6.shape)
    c4u6 = concatenate([c4, u6])    
#     print("c4u6.shape:",c4u6.shape)
#     #-----------------------------
#     c4u6u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c4u6)
# #     print("c4u6u7.shape:",c4u6u7.shape)
# #     c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(4, 4), padding='same')(c4u6)
#     c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c4u6)
#     c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c4u6u8)
# #     print("c4u6u8.shape:",c4u6u8.shape)
# #     c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(8, 8), padding='same')(c4u6)
#     c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6)
#     c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6u9)
#     c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6u9)
# #     print("c4u6u9.shape:",c4u6u9.shape)
#     #-----------------------------
    u6 = Dropout(dropout)(c4u6)
    c6 = res_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
#     print("c6.shape:",c6.shape)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
#     print("u7.shape:",u7.shape)
    c3u7 = concatenate([c3, u7])
#     c3u7 = concatenate([c3, u7,c4u6u7])
#     print("c3u7.shape:",c3u7.shape)
#     #-----------------------------
#     c3u7u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c3u7)
# #     print("c3u7u8.shape:",c3u7u8.shape)
# #     c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(4, 4), padding='same')(c3u7)
#     c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c3u7)
#     c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c3u7u9)
# #     print("c3u7u9.shape:",c3u7u9.shape)
#     #-----------------------------
    u7 = Dropout(dropout)(c3u7)
    c7 = res_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
#     print("c7.shape:",c7.shape)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
#     print("u8.shape:",u8.shape)
    c2u8 = concatenate([c2, u8])
#     c2u8 = concatenate([c2, u8,c4u6u8,c3u7u8])
# #     print("c2u8.shape:",c2u8.shape)
#     #-----------------------------
#     c2u8u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c2u8)
# #     print("c2u8u9.shape:",c2u8u9.shape)
#     #-----------------------------
    u8 = Dropout(dropout)(c2u8)
    c8 = res_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
#     print("c8.shape:",c8.shape)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
#     print("u9.shape:",u9.shape)
    c1u9 = concatenate([c1, u9], axis=3)
#     c1u9 = concatenate([c1, u9,c4u6u9,c3u7u9,c2u8u9], axis=3)
# #     print("c1u9.shape:",c1u9.shape)
#     u9 = Dropout(dropout)(c1u9)
    c9 = res_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
#     print("c9.shape:",c9.shape)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])  # 整个模型的构造
    return model
def resDunet_dense_nocc1_0912(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    
    c1 = res_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)  # 卷积
    
#     print("c1.shape:",c1.shape)
    p1 = MaxPooling2D((2, 2))(c1)  # 池化
    p1 = Dropout(dropout * 0.5)(p1)
#     print("p1.shape:",p1.shape)
    
    
    c2 = res_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
#     print("c2.shape:",c2.shape)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
#     print("p2.shape:",p2.shape)
    
    
    c3 = res_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
#     print("c3.shape:",c3.shape)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
#     print("p3.shape:",p3.shape)

    c4 = res_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
#     print("c4.shape:",c4.shape)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)
#     print("p4.shape:",p4.shape)

    c5 = res_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)
#     print("c5.shape:",c5.shape)

    # expansive path
    # 四次上采样
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
#     print("u6.shape:",u6.shape)
    c4u6 = concatenate([c4, u6])    
#     print("c4u6.shape:",c4u6.shape)
    #-----------------------------
    c4u6u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c4u6)
#     print("c4u6u7.shape:",c4u6u7.shape)
#     c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(4, 4), padding='same')(c4u6)
    c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c4u6)
    c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c4u6u8)
#     print("c4u6u8.shape:",c4u6u8.shape)
#     c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(8, 8), padding='same')(c4u6)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6u9)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6u9)
#     print("c4u6u9.shape:",c4u6u9.shape)
    #-----------------------------
    u6 = Dropout(dropout)(c4u6)
    c6 = res_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
#     print("c6.shape:",c6.shape)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
#     print("u7.shape:",u7.shape)
#     c3u7 = concatenate([c3, u7])
    c3u7 = concatenate([c3, u7,c4u6u7])
#     print("c3u7.shape:",c3u7.shape)
    #-----------------------------
    c3u7u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c3u7)
#     print("c3u7u8.shape:",c3u7u8.shape)
#     c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(4, 4), padding='same')(c3u7)
    c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c3u7)
    c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c3u7u9)
#     print("c3u7u9.shape:",c3u7u9.shape)
    #-----------------------------
    u7 = Dropout(dropout)(c3u7)
    c7 = res_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
#     print("c7.shape:",c7.shape)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
#     print("u8.shape:",u8.shape)
#     c2u8 = concatenate([c2, u8])
    c2u8 = concatenate([c2, u8,c4u6u8,c3u7u8])
#     print("c2u8.shape:",c2u8.shape)
    #-----------------------------
    c2u8u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c2u8)
#     print("c2u8u9.shape:",c2u8u9.shape)
    #-----------------------------
    u8 = Dropout(dropout)(c2u8)
    c8 = res_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
#     print("c8.shape:",c8.shape)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
#     print("u9.shape:",u9.shape)
#     c1u9 = concatenate([c1, u9], axis=3)
    c1u9 = concatenate([c1, u9,c4u6u9,c3u7u9,c2u8u9], axis=3)
#     print("c1u9.shape:",c1u9.shape)
    u9 = Dropout(dropout)(c1u9)
    c9 = res_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
#     print("c9.shape:",c9.shape)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])  # 整个模型的构造
    return model

# 20210915 res attunet  attunet（上采样用的是upsample+一个conv2d）
def up_conv(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    # keras.layers.convolutional.UpSampling2D(size=(2, 2), data_format=None)
    x = UpSampling2D(size=(2, 2))(input_tensor)

    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)

    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


# test
# test = np.random.rand(1,224,224,3)
# test = up_conv(test,64)
# print(test.shape) #(1, 448, 448, 64)


def Attention_block(inputg, inputx, F_int):
    #     print("INPUTX , inputg",inputx.shape,inputg.shape)
    W_g = Conv2D(filters=F_int, kernel_size=(1, 1), kernel_initializer="he_normal", padding="valid")(inputg)
    W_g = BatchNormalization()(W_g)
    #     print("W_g:",W_g.shape)
    W_x = Conv2D(filters=F_int, kernel_size=(1, 1), kernel_initializer="he_normal", padding="valid")(inputx)
    W_x = BatchNormalization()(W_x)
    #     print("W_x:",W_x.shape)
    # concat + relu
    psi = W_g + W_x
    psi = Activation("relu")(psi)
    psi = Conv2D(filters=1, kernel_size=(1, 1), kernel_initializer="he_normal", padding="valid")(psi)
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)
    #     print("psi:",psi.shape)
    psi = inputx * psi  # 这里是对应元素相乘 不是矩阵乘
    #     print("psi * x:",psi.shape)
    return psi


# test
# testg= np.array(np.random.rand(1,224,224,32),dtype = np.float32)
# testx= np.array(np.random.rand(1,224,224,32),dtype = np.float32)

# test = Attention_block(testg,testx,64)
# print(test.shape)
# W_g: (1, 224, 224, 64)
# W_x: (1, 224, 224, 64)
# psi: (1, 224, 224, 1)
# psi * x: (1, 224, 224, 32)
# (1, 224, 224, 32)

def res_AttU_Net_0915v1(input_img, n_filters=16, dropout=0.2, batchnorm=True):
    # print("ori_img:", input_img.shape)
    n_filters = [n_filters, n_filters * 2, n_filters * 4, n_filters * 8, n_filters * 16]

    # encoding path
    c1 = res_blockv1(input_img, n_filters[0], kernel_size=3, batchnorm=batchnorm)
    # print("c1:", c1.shape)
    p1 = MaxPooling2D((2, 2))(c1)
    # print("p1:", p1.shape)
    p1 = Dropout(dropout)(p1)

    c2 = res_blockv1(p1, n_filters[1], kernel_size=3, batchnorm=batchnorm)
    # print("c2:", c2.shape)
    p2 = MaxPooling2D((2, 2))(c2)
    # print("p2:", p2.shape)
    p2 = Dropout(dropout)(p2)

    c3 = res_blockv1(p2, n_filters[2], kernel_size=3, batchnorm=batchnorm)
    # print("c3:", c3.shape)
    p3 = MaxPooling2D((2, 2))(c3)
    # print("p3:", p3.shape)
    p3 = Dropout(dropout)(p3)

    c4 = res_blockv1(p3, n_filters[3], kernel_size=3, batchnorm=batchnorm)
    # print("c4:", c4.shape)
    p4 = MaxPooling2D((2, 2))(c4)
    # print("p4:", p4.shape)
    p4 = Dropout(dropout)(p4)

    c5 = res_blockv1(p4, n_filters[4], kernel_size=3, batchnorm=batchnorm)
    # print("c5:", c5.shape)

    # decoding + concat path
    u5 = up_conv(c5, n_filters[3], kernel_size=3, batchnorm=batchnorm)
    # print("u5:", u5.shape)
    att4 = Attention_block(inputg=u5, inputx=c4, F_int=n_filters[2])
    # print("att4:", att4.shape)
    c4u5 = concatenate([att4, u5])
    # print("cat u5 att4:", u5.shape)
#     u5 = res_blockv2(c4u5,u5, n_filters=n_filters[3], kernel_size=3, batchnorm=batchnorm)
    u5 = res_blockv1(c4u5,n_filters=n_filters[3], kernel_size=3, batchnorm=batchnorm)
    # print("u5:", u5.shape)
    
    u4 = up_conv(u5, n_filters[2], kernel_size=3, batchnorm=batchnorm)
    # print("u4:", u4.shape)
    att3 = Attention_block(inputg=u4, inputx=c3, F_int=n_filters[1])
    # print("att3:", att3.shape)
    c3u4 = concatenate([att3, u4])
    # print("cat u4 att3:", u4.shape)
    u4 = res_blockv1(c3u4,n_filters=n_filters[2], kernel_size=3, batchnorm=batchnorm)
#     u4 =  res_blockv2(c3u4,u4, n_filters=n_filters[2], kernel_size=3, batchnorm=batchnorm)
    # print("u4:", u4.shape)

    u3 = up_conv(u4, n_filters[1], kernel_size=3, batchnorm=batchnorm)
    # print("u3:", u3.shape)
    att2 = Attention_block(inputg=u3, inputx=c2, F_int=n_filters[0])
    # print("att2:", att2.shape)
    c2u3 = concatenate([att2, u3])
    # print("cat u3 att2:", u3.shape)
    u3 =  res_blockv1(c2u3,n_filters=n_filters[1], kernel_size=3, batchnorm=batchnorm)
#     u3 =  res_blockv2(c2u3,u3, n_filters=n_filters[1], kernel_size=3, batchnorm=batchnorm)
    # print("u3:", u3.shape)

    u2 = up_conv(u3, n_filters[0], kernel_size=3, batchnorm=batchnorm)
    # print("u2:", u2.shape)
    att1 = Attention_block(inputg=u2, inputx=c1, F_int=int(n_filters[0] / 2))
    # print("att1:", att1.shape)
    c1u2 = concatenate([att1, u2])
    # print("cat u2 att1:", u2.shape)
    u2 = res_blockv1(c1u2,n_filters=n_filters[0], kernel_size=3, batchnorm=batchnorm)
#     u2 = res_blockv2(c1u2,u2, n_filters=n_filters[0], kernel_size=3, batchnorm=batchnorm)
    # print("u2:", u2.shape)

    u1 = Conv2D(filters=1, kernel_size=(1, 1), kernel_initializer="he_normal", padding="valid")(u2)
    # print("u1:", u1.shape)
    u1 = Activation('sigmoid')(u1)

    outputs = u1

    model = Model(inputs=[input_img], outputs=[outputs])
    return model
def res_AttU_Net_0915v2(input_img, n_filters=16, dropout=0.2, batchnorm=True):
    # print("ori_img:", input_img.shape)
    n_filters = [n_filters, n_filters * 2, n_filters * 4, n_filters * 8, n_filters * 16]

    # encoding path
    c1 = res_blockv1(input_img, n_filters[0], kernel_size=3, batchnorm=batchnorm)
    # print("c1:", c1.shape)
    p1 = MaxPooling2D((2, 2))(c1)
    # print("p1:", p1.shape)
    p1 = Dropout(dropout)(p1)

    c2 = res_blockv1(p1, n_filters[1], kernel_size=3, batchnorm=batchnorm)
    # print("c2:", c2.shape)
    p2 = MaxPooling2D((2, 2))(c2)
    # print("p2:", p2.shape)
    p2 = Dropout(dropout)(p2)

    c3 = res_blockv1(p2, n_filters[2], kernel_size=3, batchnorm=batchnorm)
    # print("c3:", c3.shape)
    p3 = MaxPooling2D((2, 2))(c3)
    # print("p3:", p3.shape)
    p3 = Dropout(dropout)(p3)

    c4 = res_blockv1(p3, n_filters[3], kernel_size=3, batchnorm=batchnorm)
    # print("c4:", c4.shape)
    p4 = MaxPooling2D((2, 2))(c4)
    # print("p4:", p4.shape)
    p4 = Dropout(dropout)(p4)

    c5 = res_blockv1(p4, n_filters[4], kernel_size=3, batchnorm=batchnorm)
    # print("c5:", c5.shape)

    # decoding + concat path
    u5 = up_conv(c5, n_filters[3], kernel_size=3, batchnorm=batchnorm)
    # print("u5:", u5.shape)
    att4 = Attention_block(inputg=u5, inputx=c4, F_int=n_filters[2])
    # print("att4:", att4.shape)
    c4u5 = concatenate([att4, u5])
    # print("cat u5 att4:", u5.shape)
    u5 = res_blockv2(c4u5,u5, n_filters=n_filters[3], kernel_size=3, batchnorm=batchnorm)
#     u5 = res_blockv1(c4u5,n_filters=n_filters[3], kernel_size=3, batchnorm=batchnorm)
    # print("u5:", u5.shape)
    
    u4 = up_conv(u5, n_filters[2], kernel_size=3, batchnorm=batchnorm)
    # print("u4:", u4.shape)
    att3 = Attention_block(inputg=u4, inputx=c3, F_int=n_filters[1])
    # print("att3:", att3.shape)
    c3u4 = concatenate([att3, u4])
    # print("cat u4 att3:", u4.shape)
#     u4 = res_blockv1(c3u4,n_filters=n_filters[2], kernel_size=3, batchnorm=batchnorm)
    u4 =  res_blockv2(c3u4,u4, n_filters=n_filters[2], kernel_size=3, batchnorm=batchnorm)
    # print("u4:", u4.shape)

    u3 = up_conv(u4, n_filters[1], kernel_size=3, batchnorm=batchnorm)
    # print("u3:", u3.shape)
    att2 = Attention_block(inputg=u3, inputx=c2, F_int=n_filters[0])
    # print("att2:", att2.shape)
    c2u3 = concatenate([att2, u3])
    # print("cat u3 att2:", u3.shape)
#     u3 =  res_blockv1(c2u3,n_filters=n_filters[1], kernel_size=3, batchnorm=batchnorm)
    u3 =  res_blockv2(c2u3,u3, n_filters=n_filters[1], kernel_size=3, batchnorm=batchnorm)
    # print("u3:", u3.shape)

    u2 = up_conv(u3, n_filters[0], kernel_size=3, batchnorm=batchnorm)
    # print("u2:", u2.shape)
    att1 = Attention_block(inputg=u2, inputx=c1, F_int=int(n_filters[0] / 2))
    # print("att1:", att1.shape)
    c1u2 = concatenate([att1, u2])
    # print("cat u2 att1:", u2.shape)
#     u2 = res_blockv1(c1u2,n_filters=n_filters[0], kernel_size=3, batchnorm=batchnorm)
    u2 = res_blockv2(c1u2,u2, n_filters=n_filters[0], kernel_size=3, batchnorm=batchnorm)
    # print("u2:", u2.shape)

    u1 = Conv2D(filters=1, kernel_size=(1, 1), kernel_initializer="he_normal", padding="valid")(u2)
    # print("u1:", u1.shape)
    u1 = Activation('sigmoid')(u1)

    outputs = u1

    model = Model(inputs=[input_img], outputs=[outputs])
    return model
# 20210916  res att 中间加dac spp blk ==cenet + attention 的机制就行了
def SPPblockfor32(input_tensor, n_filters=1, kernel_size=1, batchnorm=True):
    # size 512 到这边 是 16   size 448到这里是 14
    # 然后这边 最后都是1通道
    # upsampling 可以直接理解为 pooling的反向操作  不需要参数的学习  conv2dtranspose才是需要参数学习的
    # 这个模块就是先进行 maxpool 然后 再upsamp 就完事了
    # 20210602
    pool1 = MaxPool2D(pool_size=(2, 2), strides=2)(input_tensor)
#     print("up1:",pool1.shape)
    conv1 = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                   padding="same")(pool1)
    up1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv1)
#     print("up1:",up1.shape)

    # 16-zeropadding-18-maxpool-6-upsamp-18-maxpool-16
    # 32-zeropadding-36-maxpool-12-upsamp-36-maxpool-32
    pool2 = ZeroPadding2D(padding=(2, 2), data_format=None)(input_tensor)
#     print("up2:",pool2.shape)
    pool2 = MaxPool2D(pool_size=(3, 3), strides=3)(pool2)
#     print("up2:",pool2.shape)
    conv2 = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                   padding="same")(pool2)
    up2 = UpSampling2D(size=(3, 3), interpolation='bilinear')(conv2)
#     print("up2:",up2.shape)
    up2 = MaxPool2D(pool_size=(5, 5), strides=1)(up2)
#     print("up2:",up2.shape)

    # 16-zeropadding-20-maxpool-4-upsamp-20-maxpool-16
    # 这里不用5x5pool 用4x4 pool就行
#     pool3 = ZeroPadding2D(padding=(2, 2), data_format=None)(input_tensor)
#     print("up3:",pool3.shape)
#     pool3 = MaxPool2D(pool_size=(5, 5), strides=5)(pool3)
#     print("up3:",pool3.shape)
#     conv3 = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
#                    padding="same")(pool3)
#     up3 = UpSampling2D(size=(5, 5), interpolation='bilinear')(conv3)
#     print("up3:",up3.shape)
#     up3 = MaxPool2D(pool_size=(5, 5), strides=1)(up3)
#     print("up3:",up3.shape)
    pool3 = MaxPool2D(pool_size=(4, 4), strides=4)(input_tensor)
#     print("up3:",pool3.shape)
    conv3 = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                   padding="same")(pool3)
    up3 = UpSampling2D(size=(4, 4), interpolation='bilinear')(conv3)
#     print("up3:",up3.shape)

    # 16-zeropadding-18-maxpool-3-upsamp-18-maxpool-16
    # 32-zeropadding-36-maxpool-6-upsamp-36-maxpool-32
    pool4 = ZeroPadding2D(padding=(2, 2), data_format=None)(input_tensor)
#     print("up4:",pool4.shape)
    pool4 = MaxPool2D(pool_size=(6, 6), strides=6)(pool4)
#     print("up4:",pool4.shape)
    conv4 = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                   padding="same")(pool4)
    up4 = UpSampling2D(size=(6, 6), interpolation='bilinear')(conv4)
#     print("up4:",up4.shape)
    up4 = MaxPool2D(pool_size=(5, 5), strides=1)(up4)
#     print("up4:",up4.shape)

    out = concatenate([up1, up2, up3, up4, input_tensor])
    return out
def res_AttU_Net_cenet_0917v2(input_img, n_filters=16, dropout=0.2, batchnorm=True):
    # print("ori_img:", input_img.shape)
    n_filters = [n_filters, n_filters * 2, n_filters * 4, n_filters * 8, n_filters * 16]

    # encoding path
    c1 = res_blockv1(input_img, n_filters[0], kernel_size=3, batchnorm=batchnorm)
    # print("c1:", c1.shape)
    p1 = MaxPooling2D((2, 2))(c1)
    # print("p1:", p1.shape)
    p1 = Dropout(dropout)(p1)

    c2 = res_blockv1(p1, n_filters[1], kernel_size=3, batchnorm=batchnorm)
    # print("c2:", c2.shape)
    p2 = MaxPooling2D((2, 2))(c2)
    # print("p2:", p2.shape)
    p2 = Dropout(dropout)(p2)

    c3 = res_blockv1(p2, n_filters[2], kernel_size=3, batchnorm=batchnorm)
    # print("c3:", c3.shape)
    p3 = MaxPooling2D((2, 2))(c3)
    # print("p3:", p3.shape)
    p3 = Dropout(dropout)(p3)

    c4 = res_blockv1(p3, n_filters[3], kernel_size=3, batchnorm=batchnorm)
    # print("c4:", c4.shape)
    p4 = MaxPooling2D((2, 2))(c4)
    # print("p4:", p4.shape)
    p4 = Dropout(dropout)(p4)

    c5 = res_blockv1(p4, n_filters[4], kernel_size=3, batchnorm=batchnorm)
    print("c5:", c5.shape)
#---------cenet blk-----
    c5 = DACblock(c5,n_filters[4])
    print("dac-c5:",c5.shape)
    c5 = SPPblockfor32(c5)
    print("spp-c5:",c5.shape)
#------------
    # decoding + concat path
    u5 = up_conv(c5, n_filters[3], kernel_size=3, batchnorm=batchnorm)
    # print("u5:", u5.shape)
    att4 = Attention_block(inputg=u5, inputx=c4, F_int=n_filters[2])
    # print("att4:", att4.shape)
    c4u5 = concatenate([att4, u5])
    # print("cat u5 att4:", u5.shape)
    u5 = res_blockv2(c4u5,u5, n_filters=n_filters[3], kernel_size=3, batchnorm=batchnorm)
#     u5 = res_blockv1(c4u5,n_filters=n_filters[3], kernel_size=3, batchnorm=batchnorm)
    # print("u5:", u5.shape)
    
    u4 = up_conv(u5, n_filters[2], kernel_size=3, batchnorm=batchnorm)
    # print("u4:", u4.shape)
    att3 = Attention_block(inputg=u4, inputx=c3, F_int=n_filters[1])
    # print("att3:", att3.shape)
    c3u4 = concatenate([att3, u4])
    # print("cat u4 att3:", u4.shape)
#     u4 = res_blockv1(c3u4,n_filters=n_filters[2], kernel_size=3, batchnorm=batchnorm)
    u4 =  res_blockv2(c3u4,u4, n_filters=n_filters[2], kernel_size=3, batchnorm=batchnorm)
    # print("u4:", u4.shape)

    u3 = up_conv(u4, n_filters[1], kernel_size=3, batchnorm=batchnorm)
    # print("u3:", u3.shape)
    att2 = Attention_block(inputg=u3, inputx=c2, F_int=n_filters[0])
    # print("att2:", att2.shape)
    c2u3 = concatenate([att2, u3])
    # print("cat u3 att2:", u3.shape)
#     u3 =  res_blockv1(c2u3,n_filters=n_filters[1], kernel_size=3, batchnorm=batchnorm)
    u3 =  res_blockv2(c2u3,u3, n_filters=n_filters[1], kernel_size=3, batchnorm=batchnorm)
    # print("u3:", u3.shape)

    u2 = up_conv(u3, n_filters[0], kernel_size=3, batchnorm=batchnorm)
    # print("u2:", u2.shape)
    att1 = Attention_block(inputg=u2, inputx=c1, F_int=int(n_filters[0] / 2))
    # print("att1:", att1.shape)
    c1u2 = concatenate([att1, u2])
    # print("cat u2 att1:", u2.shape)
#     u2 = res_blockv1(c1u2,n_filters=n_filters[0], kernel_size=3, batchnorm=batchnorm)
    u2 = res_blockv2(c1u2,u2, n_filters=n_filters[0], kernel_size=3, batchnorm=batchnorm)
    # print("u2:", u2.shape)

    u1 = Conv2D(filters=1, kernel_size=(1, 1), kernel_initializer="he_normal", padding="valid")(u2)
    # print("u1:", u1.shape)
    u1 = Activation('sigmoid')(u1)

    outputs = u1

    model = Model(inputs=[input_img], outputs=[outputs])
    return model

def res_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    #最初的v0 版本res  弄错了 
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)#批量标准化
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)#激活函数
    # 用CONV 1x1 对 inputtensor 扩维后才能和 x相加
    #  **************错误点 仍然对x 进一步的卷积********
    input_tensor = Conv2D(filters=n_filters, kernel_size=(1, 1), kernel_initializer="he_normal",
               padding="same")(x)
#     print("x and input:",x.shape,input_tensor.shape)
    res = input_tensor + x
    return res
def res_blockv1(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    #  *****把 v0 的错 改正 而已
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)#批量标准化
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)#激活函数
    # 用CONV 1x1 对 inputtensor 扩维后才能和 x相加
    input_tensor = Conv2D(filters=n_filters, kernel_size=(1, 1), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        input_tensor = BatchNormalization()(input_tensor)#批量标准化
    input_tensor = Activation("relu")(input_tensor)
#     print("x and input:",x.shape,input_tensor.shape)
    res = input_tensor + x
    return res
def res_blockv2(input_tensor, res_tensor,n_filters, kernel_size=3, batchnorm=True):
    # v2 在v1的基础上，在decorder路径上有  进来做res的是concat之前的fm feature map  而 卷积的是concat后的fm
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)#批量标准化
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)#激活函数
    # 用CONV 1x1 对 inputtensor 扩维后才能和 x相加
    res_tensor = Conv2D(filters=n_filters, kernel_size=(1, 1), kernel_initializer="he_normal",
               padding="same")(res_tensor)
    if batchnorm:
        res_tensor = BatchNormalization()(res_tensor)#批量标准化
    res_tensor = Activation("relu")(res_tensor)
#     print("x and input:",x.shape,input_tensor.shape)
    res = res_tensor + x
    return res
def resunet0912v1(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    # 四次最大池化下采样
    c1 = res_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)  # 卷积
    p1 = MaxPooling2D((2, 2))(c1)  # 池化
    p1 = Dropout(dropout * 0.5)(p1)

    c2 = res_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = res_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = res_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = res_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # expansive path
    # 四次上采样
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = res_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = res_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = res_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = res_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])  # 整个模型的构造
    return model
def resunet0912v2(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    # 四次最大池化下采样
    c1 = res_blockv1(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)  # 卷积
    p1 = MaxPooling2D((2, 2))(c1)  # 池化
    p1 = Dropout(dropout * 0.5)(p1)

    c2 = res_blockv1(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = res_blockv1(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = res_blockv1(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = res_blockv1(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # expansive path
    # 四次上采样
    #  res_blockv2(input_tensor, res_tensor,n_filters, kernel_size=3, batchnorm=True):
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    c4u6 = concatenate([u6, c4])
    c4u6 = Dropout(dropout)(c4u6)
    c6 = res_blockv2(c4u6,u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    c3u7 = concatenate([u7, c3])
    c3u7 = Dropout(dropout)(c3u7)
    c7 = res_blockv2(c3u7,u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    c2u8 = concatenate([u8, c2])
    c2u8 = Dropout(dropout)(c2u8)
    c8 = res_blockv2(c2u8,u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    c1u9 = concatenate([u9, c1], axis=3)
    c1u9 = Dropout(dropout)(c1u9)
    c9 = res_blockv2(c1u9,u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])  # 整个模型的构造
    return model
def resDunet0912(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    # 四次最大池化下采样
    cc1 = convonce(input_img, n_filters=n_filters * 2, kernel_size=3, strides=2, batchnorm=batchnorm)  # 卷积
    cc2 = convonce(cc1, n_filters=n_filters * 4, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    cc3 = convonce(cc2, n_filters=n_filters * 8, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    cc4 = convonce(cc3, n_filters=n_filters * 16, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
#     print("cc1,cc2,cc3,cc4.shape:",cc1.shape,cc2.shape,cc3.shape,cc4.shape)
    
    c1 = res_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)  # 卷积
    
#     print("c1.shape:",c1.shape)
    p1 = MaxPooling2D((2, 2))(c1)  # 池化
    p1 = Dropout(dropout * 0.5)(p1)
#     print("p1.shape:",p1.shape)
    p1 = concatenate([cc1, p1])
#     print("cc1+p1.shape:",p1.shape)
    
    
    c2 = res_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
#     print("c2.shape:",c2.shape)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
#     print("p2.shape:",p2.shape)
    p2 = concatenate([cc2, p2])
#     print("cc2+p2.shape:",p2.shape)
    
    
    c3 = res_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
#     print("c3.shape:",c3.shape)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
#     print("p3.shape:",p3.shape)
    p3 = concatenate([cc3, p3])
#     print("cc3+p3.shape:",p3.shape)

    c4 = res_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
#     print("c4.shape:",c4.shape)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)
#     print("p4.shape:",p4.shape)
    p4 = concatenate([cc4, p4])
#     print("cc4+p4.shape:",p4.shape)

    c5 = res_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)
#     print("c5.shape:",c5.shape)

    # expansive path
    # 四次上采样
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
#     print("u6.shape:",u6.shape)
    c4u6 = concatenate([c4, u6])    
#     print("c4u6.shape:",c4u6.shape)
    #-----------------------------
    c4u6u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c4u6)
#     print("c4u6u7.shape:",c4u6u7.shape)
#     c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(4, 4), padding='same')(c4u6)
    c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c4u6)
    c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c4u6u8)
#     print("c4u6u8.shape:",c4u6u8.shape)
#     c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(8, 8), padding='same')(c4u6)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6u9)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6u9)
#     print("c4u6u9.shape:",c4u6u9.shape)
    #-----------------------------
    u6 = Dropout(dropout)(c4u6)
    c6 = res_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
#     print("c6.shape:",c6.shape)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
#     print("u7.shape:",u7.shape)
#     c3u7 = concatenate([c3, u7])
    c3u7 = concatenate([c3, u7,c4u6u7])
#     print("c3u7.shape:",c3u7.shape)
    #-----------------------------
    c3u7u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c3u7)
#     print("c3u7u8.shape:",c3u7u8.shape)
#     c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(4, 4), padding='same')(c3u7)
    c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c3u7)
    c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c3u7u9)
#     print("c3u7u9.shape:",c3u7u9.shape)
    #-----------------------------
    u7 = Dropout(dropout)(c3u7)
    c7 = res_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
#     print("c7.shape:",c7.shape)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
#     print("u8.shape:",u8.shape)
#     c2u8 = concatenate([c2, u8])
    c2u8 = concatenate([c2, u8,c4u6u8,c3u7u8])
#     print("c2u8.shape:",c2u8.shape)
    #-----------------------------
    c2u8u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c2u8)
#     print("c2u8u9.shape:",c2u8u9.shape)
    #-----------------------------
    u8 = Dropout(dropout)(c2u8)
    c8 = res_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
#     print("c8.shape:",c8.shape)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
#     print("u9.shape:",u9.shape)
#     c1u9 = concatenate([c1, u9], axis=3)
    c1u9 = concatenate([c1, u9,c4u6u9,c3u7u9,c2u8u9], axis=3)
#     print("c1u9.shape:",c1u9.shape)
    u9 = Dropout(dropout)(c1u9)
    c9 = res_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
#     print("c9.shape:",c9.shape)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])  # 整个模型的构造
    return model

def resDunet_cc1_nodense_0912(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    # 四次最大池化下采样
    cc1 = convonce(input_img, n_filters=n_filters * 2, kernel_size=3, strides=2, batchnorm=batchnorm)  # 卷积
    cc2 = convonce(cc1, n_filters=n_filters * 4, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    cc3 = convonce(cc2, n_filters=n_filters * 8, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    cc4 = convonce(cc3, n_filters=n_filters * 16, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
#     print("cc1,cc2,cc3,cc4.shape:",cc1.shape,cc2.shape,cc3.shape,cc4.shape)
    
    c1 = res_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)  # 卷积
    
#     print("c1.shape:",c1.shape)
    p1 = MaxPooling2D((2, 2))(c1)  # 池化
    p1 = Dropout(dropout * 0.5)(p1)
#     print("p1.shape:",p1.shape)
    p1 = concatenate([cc1, p1])
#     print("cc1+p1.shape:",p1.shape)
    
    
    c2 = res_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
#     print("c2.shape:",c2.shape)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
#     print("p2.shape:",p2.shape)
    p2 = concatenate([cc2, p2])
#     print("cc2+p2.shape:",p2.shape)
    
    
    c3 = res_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
#     print("c3.shape:",c3.shape)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
#     print("p3.shape:",p3.shape)
    p3 = concatenate([cc3, p3])
#     print("cc3+p3.shape:",p3.shape)

    c4 = res_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
#     print("c4.shape:",c4.shape)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)
#     print("p4.shape:",p4.shape)
    p4 = concatenate([cc4, p4])
#     print("cc4+p4.shape:",p4.shape)

    c5 = res_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)
#     print("c5.shape:",c5.shape)

    # expansive path
    # 四次上采样
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
#     print("u6.shape:",u6.shape)
    c4u6 = concatenate([c4, u6])    
#     print("c4u6.shape:",c4u6.shape)
#     #-----------------------------
#     c4u6u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c4u6)
# #     print("c4u6u7.shape:",c4u6u7.shape)
# #     c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(4, 4), padding='same')(c4u6)
#     c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c4u6)
#     c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c4u6u8)
# #     print("c4u6u8.shape:",c4u6u8.shape)
# #     c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(8, 8), padding='same')(c4u6)
#     c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6)
#     c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6u9)
#     c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6u9)
# #     print("c4u6u9.shape:",c4u6u9.shape)
#     #-----------------------------
    u6 = Dropout(dropout)(c4u6)
    c6 = res_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
#     print("c6.shape:",c6.shape)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
#     print("u7.shape:",u7.shape)
    c3u7 = concatenate([c3, u7])
#     c3u7 = concatenate([c3, u7,c4u6u7])
#     print("c3u7.shape:",c3u7.shape)
#     #-----------------------------
#     c3u7u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c3u7)
# #     print("c3u7u8.shape:",c3u7u8.shape)
# #     c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(4, 4), padding='same')(c3u7)
#     c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c3u7)
#     c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c3u7u9)
# #     print("c3u7u9.shape:",c3u7u9.shape)
#     #-----------------------------
    u7 = Dropout(dropout)(c3u7)
    c7 = res_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
#     print("c7.shape:",c7.shape)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
#     print("u8.shape:",u8.shape)
    c2u8 = concatenate([c2, u8])
#     c2u8 = concatenate([c2, u8,c4u6u8,c3u7u8])
# #     print("c2u8.shape:",c2u8.shape)
#     #-----------------------------
#     c2u8u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c2u8)
# #     print("c2u8u9.shape:",c2u8u9.shape)
#     #-----------------------------
    u8 = Dropout(dropout)(c2u8)
    c8 = res_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
#     print("c8.shape:",c8.shape)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
#     print("u9.shape:",u9.shape)
    c1u9 = concatenate([c1, u9], axis=3)
#     c1u9 = concatenate([c1, u9,c4u6u9,c3u7u9,c2u8u9], axis=3)
# #     print("c1u9.shape:",c1u9.shape)
#     u9 = Dropout(dropout)(c1u9)
    c9 = res_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
#     print("c9.shape:",c9.shape)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])  # 整个模型的构造
    return model
def resDunet_dense_nocc1_0912(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    
    c1 = res_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)  # 卷积
    
#     print("c1.shape:",c1.shape)
    p1 = MaxPooling2D((2, 2))(c1)  # 池化
    p1 = Dropout(dropout * 0.5)(p1)
#     print("p1.shape:",p1.shape)
    
    
    c2 = res_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
#     print("c2.shape:",c2.shape)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
#     print("p2.shape:",p2.shape)
    
    
    c3 = res_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
#     print("c3.shape:",c3.shape)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
#     print("p3.shape:",p3.shape)

    c4 = res_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
#     print("c4.shape:",c4.shape)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)
#     print("p4.shape:",p4.shape)

    c5 = res_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)
#     print("c5.shape:",c5.shape)

    # expansive path
    # 四次上采样
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
#     print("u6.shape:",u6.shape)
    c4u6 = concatenate([c4, u6])    
#     print("c4u6.shape:",c4u6.shape)
    #-----------------------------
    c4u6u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c4u6)
#     print("c4u6u7.shape:",c4u6u7.shape)
#     c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(4, 4), padding='same')(c4u6)
    c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c4u6)
    c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c4u6u8)
#     print("c4u6u8.shape:",c4u6u8.shape)
#     c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(8, 8), padding='same')(c4u6)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6u9)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6u9)
#     print("c4u6u9.shape:",c4u6u9.shape)
    #-----------------------------
    u6 = Dropout(dropout)(c4u6)
    c6 = res_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
#     print("c6.shape:",c6.shape)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
#     print("u7.shape:",u7.shape)
#     c3u7 = concatenate([c3, u7])
    c3u7 = concatenate([c3, u7,c4u6u7])
#     print("c3u7.shape:",c3u7.shape)
    #-----------------------------
    c3u7u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c3u7)
#     print("c3u7u8.shape:",c3u7u8.shape)
#     c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(4, 4), padding='same')(c3u7)
    c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c3u7)
    c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c3u7u9)
#     print("c3u7u9.shape:",c3u7u9.shape)
    #-----------------------------
    u7 = Dropout(dropout)(c3u7)
    c7 = res_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
#     print("c7.shape:",c7.shape)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
#     print("u8.shape:",u8.shape)
#     c2u8 = concatenate([c2, u8])
    c2u8 = concatenate([c2, u8,c4u6u8,c3u7u8])
#     print("c2u8.shape:",c2u8.shape)
    #-----------------------------
    c2u8u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c2u8)
#     print("c2u8u9.shape:",c2u8u9.shape)
    #-----------------------------
    u8 = Dropout(dropout)(c2u8)
    c8 = res_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
#     print("c8.shape:",c8.shape)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
#     print("u9.shape:",u9.shape)
#     c1u9 = concatenate([c1, u9], axis=3)
    c1u9 = concatenate([c1, u9,c4u6u9,c3u7u9,c2u8u9], axis=3)
#     print("c1u9.shape:",c1u9.shape)
    u9 = Dropout(dropout)(c1u9)
    c9 = res_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
#     print("c9.shape:",c9.shape)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])  # 整个模型的构造
    return model
def get_unet_withcc1_only0908(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    # 四次最大池化下采样
    cc1 = convonce(input_img, n_filters=n_filters * 2, kernel_size=3, strides=2, batchnorm=batchnorm)  # 卷积
    cc2 = convonce(cc1, n_filters=n_filters * 4, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    cc3 = convonce(cc2, n_filters=n_filters * 8, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    cc4 = convonce(cc3, n_filters=n_filters * 16, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
#     print("cc1,cc2,cc3,cc4.shape:",cc1.shape,cc2.shape,cc3.shape,cc4.shape)
    
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)  # 卷积
    
#     print("c1.shape:",c1.shape)
    p1 = MaxPooling2D((2, 2))(c1)  # 池化
    p1 = Dropout(dropout * 0.5)(p1)
#     print("p1.shape:",p1.shape)
    p1 = concatenate([cc1, p1])
#     print("cc1+p1.shape:",p1.shape)
    
    
    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
#     print("c2.shape:",c2.shape)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
#     print("p2.shape:",p2.shape)
    p2 = concatenate([cc2, p2])
#     print("cc2+p2.shape:",p2.shape)
    
    
    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
#     print("c3.shape:",c3.shape)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
#     print("p3.shape:",p3.shape)
    p3 = concatenate([cc3, p3])
#     print("cc3+p3.shape:",p3.shape)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
#     print("c4.shape:",c4.shape)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)
#     print("p4.shape:",p4.shape)
    p4 = concatenate([cc4, p4])
#     print("cc4+p4.shape:",p4.shape)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)
#     print("c5.shape:",c5.shape)

    # expansive path
    # 四次上采样
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
#     print("u6.shape:",u6.shape)
    c4u6 = concatenate([c4, u6])    
#     print("c4u6.shape:",c4u6.shape)
    #-----------------------------
#     print("c4u6u9.shape:",c4u6u9.shape)
    #-----------------------------
    u6 = Dropout(dropout)(c4u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
#     print("c6.shape:",c6.shape)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
#     print("u7.shape:",u7.shape)
#     c3u7 = concatenate([c3, u7])
    c3u7 = concatenate([c3, u7])
#     print("c3u7.shape:",c3u7.shape)
    #-----------------------------
#     print("c3u7u9.shape:",c3u7u9.shape)
    #-----------------------------
    u7 = Dropout(dropout)(c3u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
#     print("c7.shape:",c7.shape)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
#     print("u8.shape:",u8.shape)
#     c2u8 = concatenate([c2, u8])
    c2u8 = concatenate([c2, u8])
#     print("c2u8.shape:",c2u8.shape)
    #-----------------------------
  
    #-----------------------------
    u8 = Dropout(dropout)(c2u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
#     print("c8.shape:",c8.shape)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
#     print("u9.shape:",u9.shape)
#     c1u9 = concatenate([c1, u9], axis=3)
    c1u9 = concatenate([c1, u9], axis=3)
#     print("c1u9.shape:",c1u9.shape)
    u9 = Dropout(dropout)(c1u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
#     print("c9.shape:",c9.shape)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])  # 整个模型的构造
    return model
def get_unet_withdenseonly_0908(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)  # 卷积
    
#     print("c1.shape:",c1.shape)
    p1 = MaxPooling2D((2, 2))(c1)  # 池化
    p1 = Dropout(dropout * 0.5)(p1)
#     print("p1.shape:",p1.shape)
    
    
    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
#     print("c2.shape:",c2.shape)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
#     print("p2.shape:",p2.shape)
    
    
    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
#     print("c3.shape:",c3.shape)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
#     print("p3.shape:",p3.shape)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
#     print("c4.shape:",c4.shape)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)
#     print("p4.shape:",p4.shape)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)
#     print("c5.shape:",c5.shape)

    # expansive path
    # 四次上采样
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
#     print("u6.shape:",u6.shape)
    c4u6 = concatenate([c4, u6])    
#     print("c4u6.shape:",c4u6.shape)
    #-----------------------------
    c4u6u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c4u6)
#     print("c4u6u7.shape:",c4u6u7.shape)
#     c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(4, 4), padding='same')(c4u6)
    c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c4u6)
    c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c4u6u8)
#     print("c4u6u8.shape:",c4u6u8.shape)
#     c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(8, 8), padding='same')(c4u6)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6u9)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6u9)
#     print("c4u6u9.shape:",c4u6u9.shape)
    #-----------------------------
    u6 = Dropout(dropout)(c4u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
#     print("c6.shape:",c6.shape)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
#     print("u7.shape:",u7.shape)
#     c3u7 = concatenate([c3, u7])
    c3u7 = concatenate([c3, u7,c4u6u7])
#     print("c3u7.shape:",c3u7.shape)
    #-----------------------------
    c3u7u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c3u7)
#     print("c3u7u8.shape:",c3u7u8.shape)
#     c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(4, 4), padding='same')(c3u7)
    c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c3u7)
    c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c3u7u9)
#     print("c3u7u9.shape:",c3u7u9.shape)
    #-----------------------------
    u7 = Dropout(dropout)(c3u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
#     print("c7.shape:",c7.shape)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
#     print("u8.shape:",u8.shape)
#     c2u8 = concatenate([c2, u8])
    c2u8 = concatenate([c2, u8,c4u6u8,c3u7u8])
#     print("c2u8.shape:",c2u8.shape)
    #-----------------------------
    c2u8u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c2u8)
#     print("c2u8u9.shape:",c2u8u9.shape)
    #-----------------------------
    u8 = Dropout(dropout)(c2u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
#     print("c8.shape:",c8.shape)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
#     print("u9.shape:",u9.shape)
#     c1u9 = concatenate([c1, u9], axis=3)
    c1u9 = concatenate([c1, u9,c4u6u9,c3u7u9,c2u8u9], axis=3)
#     print("c1u9.shape:",c1u9.shape)
    u9 = Dropout(dropout)(c1u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
#     print("c9.shape:",c9.shape)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])  # 整个模型的构造
    return model
def att_unet_with_cc1_print(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    #不加dense模块 就平常的att+前面的cc1
    # contracting path
    # 四次最大池化下采样
    cc1 = convonce(input_img, n_filters=n_filters * 2, kernel_size=3, strides=2, batchnorm=batchnorm)  # 卷积
    cc2 = convonce(cc1, n_filters=n_filters * 4, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    cc3 = convonce(cc2, n_filters=n_filters * 8, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    cc4 = convonce(cc3, n_filters=n_filters * 16, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    print("cc1,cc2,cc3,cc4.shape:",cc1.shape,cc2.shape,cc3.shape,cc4.shape)
    
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)  # 卷积
    
    print("c1.shape:",c1.shape)
    p1 = MaxPooling2D((2, 2))(c1)  # 池化
    p1 = Dropout(dropout * 0.5)(p1)
    print("p1.shape:",p1.shape)
    p1 = concatenate([cc1, p1])
    print("cc1+p1.shape:",p1.shape)
    
    
    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    print("c2.shape:",c2.shape)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    print("p2.shape:",p2.shape)
    p2 = concatenate([cc2, p2])
    print("cc2+p2.shape:",p2.shape)
    
    
    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    print("c3.shape:",c3.shape)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    print("p3.shape:",p3.shape)
    p3 = concatenate([cc3, p3])
    print("cc3+p3.shape:",p3.shape)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    print("c4.shape:",c4.shape)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    print("p4.shape:",p4.shape)
    p4 = concatenate([cc4, p4])
    print("cc4+p4.shape:",p4.shape)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)
    print("c5.shape:",c5.shape)

    # expansive path
    # 四次上采样
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    print("u6.shape:",u6.shape)
    #---------------------att blk----
    att4 = Attention_block(inputg=u6, inputx=c4, F_int=n_filters * 4)
    print("att4=u6+c4:",att4.shape)
    #---------------------------------
    c4u6 = concatenate([att4, u6])    
    print("attc4-c4u6.shape:",c4u6.shape)
    #-----------------------------
    #-----------------------------
    u6 = Dropout(dropout)(c4u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    print("c6.shape:",c6.shape)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    print("u7.shape:",u7.shape)
    #---------------------att blk----
    att3 = Attention_block(inputg=u7, inputx=c3, F_int=n_filters * 2)
    print("att3=u7+c3:",att3.shape)
    #---------------------------------
#     c3u7 = concatenate([c3, u7])
    c3u7 = concatenate([att3, u7])
    print("attc3-c3u7.shape:",c3u7.shape)
    #-----------------------------
    #-----------------------------
    u7 = Dropout(dropout)(c3u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    print("c7.shape:",c7.shape)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    print("u8.shape:",u8.shape) 
    #---------------------att blk----
    att2 = Attention_block(inputg=u8, inputx=c2, F_int=n_filters * 1)
    print("att2=u8+c2:",att2.shape)
    #---------------------------------
    #-----------------------------
    c2u8 = concatenate([c2, u8])
    print("attc2-c2u8.shape:",c2u8.shape)
    #-----------------------------
    #-----------------------------
    u8 = Dropout(dropout)(c2u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    print("c8.shape:",c8.shape)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    print("u9.shape:",u9.shape)
    #---------------------att blk----
    att1 = Attention_block(inputg=u9, inputx=c1, F_int=n_filters /2)
    print("att1=u9+c1:",att1.shape)
    #---------------------------------
#     c1u9 = concatenate([c1, u9], axis=3)
    c1u9 = concatenate([att1, u9], axis=3)
    print("attc1-c1u9.shape:",c1u9.shape)
    u9 = Dropout(dropout)(c1u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    print("c9.shape:",c9.shape)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])  # 整个模型的构造
    return model
def att_unet_with_cc1(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    #不加dense模块 就平常的att+前面的cc1
    # contracting path
    # 四次最大池化下采样
    cc1 = convonce(input_img, n_filters=n_filters * 2, kernel_size=3, strides=2, batchnorm=batchnorm)  # 卷积
    cc2 = convonce(cc1, n_filters=n_filters * 4, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    cc3 = convonce(cc2, n_filters=n_filters * 8, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    cc4 = convonce(cc3, n_filters=n_filters * 16, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
#     print("cc1,cc2,cc3,cc4.shape:",cc1.shape,cc2.shape,cc3.shape,cc4.shape)
    
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)  # 卷积
    
#     print("c1.shape:",c1.shape)
    p1 = MaxPooling2D((2, 2))(c1)  # 池化
    p1 = Dropout(dropout * 0.5)(p1)
#     print("p1.shape:",p1.shape)
    p1 = concatenate([cc1, p1])
#     print("cc1+p1.shape:",p1.shape)
    
    
    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
#     print("c2.shape:",c2.shape)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
#     print("p2.shape:",p2.shape)
    p2 = concatenate([cc2, p2])
#     print("cc2+p2.shape:",p2.shape)
    
    
    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
#     print("c3.shape:",c3.shape)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
#     print("p3.shape:",p3.shape)
    p3 = concatenate([cc3, p3])
#     print("cc3+p3.shape:",p3.shape)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
#     print("c4.shape:",c4.shape)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)
#     print("p4.shape:",p4.shape)
    p4 = concatenate([cc4, p4])
#     print("cc4+p4.shape:",p4.shape)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)
#     print("c5.shape:",c5.shape)

    # expansive path
    # 四次上采样
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
#     print("u6.shape:",u6.shape)
    #---------------------att blk----
    att4 = Attention_block(inputg=u6, inputx=c4, F_int=n_filters * 4)
#     print("att4=u6+c4:",att4.shape)
    #---------------------------------
    c4u6 = concatenate([att4, u6])    
#     print("attc4-c4u6.shape:",c4u6.shape)
    #-----------------------------
    #-----------------------------
    u6 = Dropout(dropout)(c4u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
#     print("c6.shape:",c6.shape)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
#     print("u7.shape:",u7.shape)
    #---------------------att blk----
    att3 = Attention_block(inputg=u7, inputx=c3, F_int=n_filters * 2)
#     print("att3=u7+c3:",att3.shape)
    #---------------------------------
#     c3u7 = concatenate([c3, u7])
    c3u7 = concatenate([att3, u7])
#     print("attc3-c3u7.shape:",c3u7.shape)
    #-----------------------------
    #-----------------------------
    u7 = Dropout(dropout)(c3u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
#     print("c7.shape:",c7.shape)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
#     print("u8.shape:",u8.shape) 
    #---------------------att blk----
    att2 = Attention_block(inputg=u8, inputx=c2, F_int=n_filters * 1)
#     print("att2=u8+c2:",att2.shape)
    #---------------------------------
    #-----------------------------
    c2u8 = concatenate([c2, u8])
#     print("attc2-c2u8.shape:",c2u8.shape)
    #-----------------------------
    #-----------------------------
    u8 = Dropout(dropout)(c2u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
#     print("c8.shape:",c8.shape)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
#     print("u9.shape:",u9.shape)
    #---------------------att blk----
    att1 = Attention_block(inputg=u9, inputx=c1, F_int=n_filters /2)
#     print("att1=u9+c1:",att1.shape)
    #---------------------------------
#     c1u9 = concatenate([c1, u9], axis=3)
    c1u9 = concatenate([att1, u9], axis=3)
#     print("attc1-c1u9.shape:",c1u9.shape)
    u9 = Dropout(dropout)(c1u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
#     print("c9.shape:",c9.shape)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])  # 整个模型的构造
    return model
def conv_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        # BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def up_conv(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    # keras.layers.convolutional.UpSampling2D(size=(2, 2), data_format=None)
    x = UpSampling2D(size=(2, 2))(input_tensor)

    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)

    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x
def Attention_block(inputg, inputx, F_int):
    #     print("INPUTX , inputg",inputx.shape,inputg.shape)
    W_g = Conv2D(filters=F_int, kernel_size=(1, 1), kernel_initializer="he_normal", padding="valid")(inputg)
    W_g = BatchNormalization()(W_g)
#     print("W_g:",W_g.shape)
    W_x = Conv2D(filters=F_int, kernel_size=(1, 1), kernel_initializer="he_normal", padding="valid")(inputx)
    W_x = BatchNormalization()(W_x)
#     print("W_x:",W_x.shape)
    # concat + relu
    psi = W_g + W_x
    psi = Activation("relu")(psi)
    psi = Conv2D(filters=1, kernel_size=(1, 1), kernel_initializer="he_normal", padding="valid")(psi)
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)
#     print("psi:",psi.shape)
    psi = inputx * psi  # 这里是对应元素相乘 不是矩阵乘
#     print("psi * x:",psi.shape)
    return psi

# 效果没有v1版本的好 也就是加了 att机制并没有不加的好
def get_unet_att_convt0910(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    # 四次最大池化下采样
    cc1 = convonce(input_img, n_filters=n_filters * 2, kernel_size=3, strides=2, batchnorm=batchnorm)  # 卷积
    cc2 = convonce(cc1, n_filters=n_filters * 4, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    cc3 = convonce(cc2, n_filters=n_filters * 8, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    cc4 = convonce(cc3, n_filters=n_filters * 16, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
#     print("cc1,cc2,cc3,cc4.shape:",cc1.shape,cc2.shape,cc3.shape,cc4.shape)
    
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)  # 卷积
    
#     print("c1.shape:",c1.shape)
    p1 = MaxPooling2D((2, 2))(c1)  # 池化
    p1 = Dropout(dropout * 0.5)(p1)
#     print("p1.shape:",p1.shape)
    p1 = concatenate([cc1, p1])
#     print("cc1+p1.shape:",p1.shape)
    
    
    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
#     print("c2.shape:",c2.shape)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
#     print("p2.shape:",p2.shape)
    p2 = concatenate([cc2, p2])
#     print("cc2+p2.shape:",p2.shape)
    
    
    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
#     print("c3.shape:",c3.shape)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
#     print("p3.shape:",p3.shape)
    p3 = concatenate([cc3, p3])
#     print("cc3+p3.shape:",p3.shape)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
#     print("c4.shape:",c4.shape)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)
#     print("p4.shape:",p4.shape)
    p4 = concatenate([cc4, p4])
#     print("cc4+p4.shape:",p4.shape)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)
#     print("c5.shape:",c5.shape)

    # expansive path
    # 四次上采样
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
#     print("u6.shape:",u6.shape)
    #---------------------att blk----
    att4 = Attention_block(inputg=u6, inputx=c4, F_int=n_filters * 4)
#     print("att4=u6+c4:",att4.shape)
    #---------------------------------
    c4u6 = concatenate([att4, u6])    
#     print("attc4-c4u6.shape:",c4u6.shape)
    #-----------------------------
    c4u6u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c4u6)
#     print("c4u6u7.shape:",c4u6u7.shape)
#     c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(4, 4), padding='same')(c4u6)
    c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c4u6)
    c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c4u6u8)
#     print("c4u6u8.shape:",c4u6u8.shape)
#     c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(8, 8), padding='same')(c4u6)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6u9)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6u9)
#     print("c4u6u9.shape:",c4u6u9.shape)
    #-----------------------------
    u6 = Dropout(dropout)(c4u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
#     print("c6.shape:",c6.shape)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
#     print("u7.shape:",u7.shape)
    #---------------------att blk----
    att3 = Attention_block(inputg=u7, inputx=c3, F_int=n_filters * 2)
#     print("att3=u7+c3:",att3.shape)
    #---------------------------------
#     c3u7 = concatenate([c3, u7])
    c3u7 = concatenate([att3, u7,c4u6u7])
#     print("attc3-c3u7.shape:",c3u7.shape)
    #-----------------------------
    c3u7u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c3u7)
#     print("c3u7u8.shape:",c3u7u8.shape)
#     c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(4, 4), padding='same')(c3u7)
    c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c3u7)
    c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c3u7u9)
#     print("c3u7u9.shape:",c3u7u9.shape)
    #-----------------------------
    u7 = Dropout(dropout)(c3u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
#     print("c7.shape:",c7.shape)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
#     print("u8.shape:",u8.shape) 
    #---------------------att blk----
    att2 = Attention_block(inputg=u8, inputx=c2, F_int=n_filters * 1)
#     print("att2=u8+c2:",att2.shape)
    #---------------------------------
    #-----------------------------
#     c2u8 = concatenate([c2, u8])
    c2u8 = concatenate([att2, u8,c4u6u8,c3u7u8])
#     print("attc2-c2u8.shape:",c2u8.shape)
    #-----------------------------
    c2u8u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c2u8)
#     print("c2u8u9.shape:",c2u8u9.shape)
    #-----------------------------
    u8 = Dropout(dropout)(c2u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
#     print("c8.shape:",c8.shape)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
#     print("u9.shape:",u9.shape)
    #---------------------att blk----
    att1 = Attention_block(inputg=u9, inputx=c1, F_int=n_filters /2)
#     print("att1=u9+c1:",att2.shape)
    #---------------------------------
#     c1u9 = concatenate([c1, u9], axis=3)
    c1u9 = concatenate([att1, u9,c4u6u9,c3u7u9,c2u8u9], axis=3)
#     print("attc2-c2u8.shape:",c1u9.shape)
    u9 = Dropout(dropout)(c1u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
#     print("c9.shape:",c9.shape)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])  # 整个模型的构造
    return model
def get_unet_att_convt0910print(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    # 四次最大池化下采样
    cc1 = convonce(input_img, n_filters=n_filters * 2, kernel_size=3, strides=2, batchnorm=batchnorm)  # 卷积
    cc2 = convonce(cc1, n_filters=n_filters * 4, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    cc3 = convonce(cc2, n_filters=n_filters * 8, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    cc4 = convonce(cc3, n_filters=n_filters * 16, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    print("cc1,cc2,cc3,cc4.shape:",cc1.shape,cc2.shape,cc3.shape,cc4.shape)
    
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)  # 卷积
    
    print("c1.shape:",c1.shape)
    p1 = MaxPooling2D((2, 2))(c1)  # 池化
    p1 = Dropout(dropout * 0.5)(p1)
    print("p1.shape:",p1.shape)
    p1 = concatenate([cc1, p1])
    print("cc1+p1.shape:",p1.shape)
    
    
    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    print("c2.shape:",c2.shape)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    print("p2.shape:",p2.shape)
    p2 = concatenate([cc2, p2])
    print("cc2+p2.shape:",p2.shape)
    
    
    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    print("c3.shape:",c3.shape)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    print("p3.shape:",p3.shape)
    p3 = concatenate([cc3, p3])
    print("cc3+p3.shape:",p3.shape)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    print("c4.shape:",c4.shape)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    print("p4.shape:",p4.shape)
    p4 = concatenate([cc4, p4])
    print("cc4+p4.shape:",p4.shape)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)
    print("c5.shape:",c5.shape)

    # expansive path
    # 四次上采样
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    print("u6.shape:",u6.shape)
    #---------------------att blk----
    att4 = Attention_block(inputg=u6, inputx=c4, F_int=n_filters * 4)
    print("att4=u6+c4:",att4.shape)
    #---------------------------------
    c4u6 = concatenate([att4, u6])    
    print("attc4-c4u6.shape:",c4u6.shape)
    #-----------------------------
    c4u6u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c4u6)
    print("c4u6u7.shape:",c4u6u7.shape)
#     c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(4, 4), padding='same')(c4u6)
    c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c4u6)
    c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c4u6u8)
    print("c4u6u8.shape:",c4u6u8.shape)
#     c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(8, 8), padding='same')(c4u6)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6u9)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6u9)
    print("c4u6u9.shape:",c4u6u9.shape)
    #-----------------------------
    u6 = Dropout(dropout)(c4u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    print("c6.shape:",c6.shape)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    print("u7.shape:",u7.shape)
    #---------------------att blk----
    att3 = Attention_block(inputg=u7, inputx=c3, F_int=n_filters * 2)
    print("att3=u7+c3:",att3.shape)
    #---------------------------------
#     c3u7 = concatenate([c3, u7])
    c3u7 = concatenate([att3, u7,c4u6u7])
    print("attc3-c3u7.shape:",c3u7.shape)
    #-----------------------------
    c3u7u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c3u7)
    print("c3u7u8.shape:",c3u7u8.shape)
#     c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(4, 4), padding='same')(c3u7)
    c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c3u7)
    c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c3u7u9)
    print("c3u7u9.shape:",c3u7u9.shape)
    #-----------------------------
    u7 = Dropout(dropout)(c3u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    print("c7.shape:",c7.shape)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    print("u8.shape:",u8.shape) 
    #---------------------att blk----
    att2 = Attention_block(inputg=u8, inputx=c2, F_int=n_filters * 1)
    print("att2=u8+c2:",att2.shape)
    #---------------------------------
    #-----------------------------
#     c2u8 = concatenate([c2, u8])
    c2u8 = concatenate([att2, u8,c4u6u8,c3u7u8])
    print("attc2-c2u8.shape:",c2u8.shape)
    #-----------------------------
    c2u8u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c2u8)
    print("c2u8u9.shape:",c2u8u9.shape)
    #-----------------------------
    u8 = Dropout(dropout)(c2u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    print("c8.shape:",c8.shape)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    print("u9.shape:",u9.shape)
    #---------------------att blk----
    att1 = Attention_block(inputg=u9, inputx=c1, F_int=n_filters /2)
    print("att1=u9+c1:",att2.shape)
    #---------------------------------
#     c1u9 = concatenate([c1, u9], axis=3)
    c1u9 = concatenate([att1, u9,c4u6u9,c3u7u9,c2u8u9], axis=3)
    print("attc2-c2u8.shape:",c1u9.shape)
    u9 = Dropout(dropout)(c1u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    print("c9.shape:",c9.shape)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])  # 整个模型的构造
    return model
#20210918   res+att+Dunetv2  res+att+Dunetv2+ce  这里att上采样用convtrans
def resv2_dunet_att_0918(input_img, n_filters=16, dropout=0.5, batchnorm=True,printstate=False):
    # contracting path
    # 四次最大池化下采样
    cc1 = convonce(input_img, n_filters=n_filters * 2, kernel_size=3, strides=2, batchnorm=batchnorm)  # 卷积
    cc2 = convonce(cc1, n_filters=n_filters * 4, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    cc3 = convonce(cc2, n_filters=n_filters * 8, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    cc4 = convonce(cc3, n_filters=n_filters * 16, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    if printstate == True:
        print("cc1,cc2,cc3,cc4.shape:",cc1.shape,cc2.shape,cc3.shape,cc4.shape)
    
    c1 = res_blockv1(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)  # 卷积
    if printstate == True:
        print("c1.shape:",c1.shape)
    p1 = MaxPooling2D((2, 2))(c1)  # 池化
    p1 = Dropout(dropout * 0.5)(p1)
    if printstate == True:
        print("p1.shape:",p1.shape)
    p1 = concatenate([cc1, p1])
    if printstate == True:
        print("cc1+p1.shape:",p1.shape)
    
    
    c2 = res_blockv1(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    if printstate == True:
        print("c2.shape:",c2.shape)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    if printstate == True:
        print("p2.shape:",p2.shape)
    p2 = concatenate([cc2, p2])
    if printstate == True:
        print("cc2+p2.shape:",p2.shape)
    
    
    c3 = res_blockv1(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    if printstate == True:
        print("c3.shape:",c3.shape)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    if printstate == True:
        print("p3.shape:",p3.shape)
    p3 = concatenate([cc3, p3])
    if printstate == True:
        print("cc3+p3.shape:",p3.shape)

    c4 = res_blockv1(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    if printstate == True:
        print("c4.shape:",c4.shape)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    if printstate == True:
        print("p4.shape:",p4.shape)
    p4 = concatenate([cc4, p4])
    if printstate == True:
        print("cc4+p4.shape:",p4.shape)

    c5 = res_blockv1(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)
    if printstate == True:
        print("c5.shape:",c5.shape)

    # expansive path
    # 四次上采样
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    if printstate == True:
        print("u6.shape:",u6.shape)
    #---------------------att blk----
    att4 = Attention_block(inputg=u6, inputx=c4, F_int=n_filters * 4)
    if printstate == True:
        print("att4=u6+c4:",att4.shape)
    #---------------------------------
    c4u6 = concatenate([att4, u6])
    if printstate == True:    
        print("attc4-c4u6.shape:",c4u6.shape)
    #-----------------------------
    c4u6u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c4u6)
    if printstate == True:
        print("c4u6u7.shape:",c4u6u7.shape)
#     c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(4, 4), padding='same')(c4u6)
    c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c4u6)
    c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c4u6u8)
    if printstate == True:
        print("c4u6u8.shape:",c4u6u8.shape)
#     c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(8, 8), padding='same')(c4u6)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6u9)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6u9)
    if printstate == True:
        print("c4u6u9.shape:",c4u6u9.shape)
    #-----------------------------
    c4u6 = Dropout(dropout)(c4u6)
    c6 = res_blockv2(c4u6,u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    if printstate == True:
        print("c6.shape:",c6.shape)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    if printstate == True:
        print("u7.shape:",u7.shape)
    #---------------------att blk----
    att3 = Attention_block(inputg=u7, inputx=c3, F_int=n_filters * 2)
    if printstate == True:
        print("att3=u7+c3:",att3.shape)
    #---------------------------------
    c3u7 = concatenate([att3, u7,c4u6u7])
    if printstate == True:
        print("attc3-c3u7.shape:",c3u7.shape)
    #-----------------------------
    c3u7u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c3u7)
    if printstate == True:
        print("c3u7u8.shape:",c3u7u8.shape)
#     c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(4, 4), padding='same')(c3u7)
    c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c3u7)
    c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c3u7u9)
    if printstate == True:
        print("c3u7u9.shape:",c3u7u9.shape)
    #-----------------------------
    c3u7 = Dropout(dropout)(c3u7)
    c7 = res_blockv2(c3u7,u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    if printstate == True:
        print("c7.shape:",c7.shape)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    if printstate == True:
        print("u8.shape:",u8.shape) 
    #---------------------att blk----
    att2 = Attention_block(inputg=u8, inputx=c2, F_int=n_filters * 1)
    if printstate == True:
        print("att2=u8+c2:",att2.shape)
    #---------------------------------
    #-----------------------------
#     c2u8 = concatenate([c2, u8])
    c2u8 = concatenate([att2, u8,c4u6u8,c3u7u8])
    if printstate == True:
        print("attc2-c2u8.shape:",c2u8.shape)
    #-----------------------------
    c2u8u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c2u8)
    if printstate == True:
        print("c2u8u9.shape:",c2u8u9.shape)
    #-----------------------------
    c2u8 = Dropout(dropout)(c2u8)
    c8 = res_blockv2(c2u8,u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    if printstate == True:
        print("c8.shape:",c8.shape)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    if printstate == True:
        print("u9.shape:",u9.shape)
    #---------------------att blk----
    att1 = Attention_block(inputg=u9, inputx=c1, F_int=n_filters /2)
    if printstate == True:
        print("att1=u9+c1:",att2.shape)
    #---------------------------------
#     c1u9 = concatenate([c1, u9], axis=3)
    c1u9 = concatenate([att1, u9,c4u6u9,c3u7u9,c2u8u9], axis=3)
    if printstate == True:
        print("attc2-c2u8.shape:",c1u9.shape)
    c1u9 = Dropout(dropout)(c1u9)
    c9 = res_blockv2(c1u9,u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    if printstate == True:
        print("c9.shape:",c9.shape)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])  # 整个模型的构造
    return model
def resv2_dunet_att_ce_0918(input_img, n_filters=16, dropout=0.5, batchnorm=True,printstate=False):
    # contracting path
    # 四次最大池化下采样
    cc1 = convonce(input_img, n_filters=n_filters * 2, kernel_size=3, strides=2, batchnorm=batchnorm)  # 卷积
    cc2 = convonce(cc1, n_filters=n_filters * 4, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    cc3 = convonce(cc2, n_filters=n_filters * 8, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    cc4 = convonce(cc3, n_filters=n_filters * 16, kernel_size=3,strides=2, batchnorm=batchnorm)  # 卷积
    if printstate == True:
        print("cc1,cc2,cc3,cc4.shape:",cc1.shape,cc2.shape,cc3.shape,cc4.shape)
    
    c1 = res_blockv1(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)  # 卷积
    if printstate == True:
        print("c1.shape:",c1.shape)
    p1 = MaxPooling2D((2, 2))(c1)  # 池化
    p1 = Dropout(dropout * 0.5)(p1)
    if printstate == True:
        print("p1.shape:",p1.shape)
    p1 = concatenate([cc1, p1])
    if printstate == True:
        print("cc1+p1.shape:",p1.shape)
    
    
    c2 = res_blockv1(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    if printstate == True:
        print("c2.shape:",c2.shape)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    if printstate == True:
        print("p2.shape:",p2.shape)
    p2 = concatenate([cc2, p2])
    if printstate == True:
        print("cc2+p2.shape:",p2.shape)
    
    
    c3 = res_blockv1(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    if printstate == True:
        print("c3.shape:",c3.shape)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    if printstate == True:
        print("p3.shape:",p3.shape)
    p3 = concatenate([cc3, p3])
    if printstate == True:
        print("cc3+p3.shape:",p3.shape)

    c4 = res_blockv1(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    if printstate == True:
        print("c4.shape:",c4.shape)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    if printstate == True:
        print("p4.shape:",p4.shape)
    p4 = concatenate([cc4, p4])
    if printstate == True:
        print("cc4+p4.shape:",p4.shape)

    c5 = res_blockv1(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)
    if printstate == True:
        print("c5.shape:",c5.shape)
#---------cenet blk-----
    c5 = DACblock(c5,n_filters * 16)
    if printstate == True:
        print("dac-c5:",c5.shape)
    c5 = SPPblockfor32(c5)
    if printstate == True:
        print("spp-c5:",c5.shape)
#------------
    # expansive path
    # 四次上采样
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    if printstate == True:
        print("u6.shape:",u6.shape)
    #---------------------att blk----
    att4 = Attention_block(inputg=u6, inputx=c4, F_int=n_filters * 4)
    if printstate == True:
        print("att4=u6+c4:",att4.shape)
    #---------------------------------
    c4u6 = concatenate([att4, u6])
    if printstate == True:    
        print("attc4-c4u6.shape:",c4u6.shape)
    #-----------------------------
    c4u6u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c4u6)
    if printstate == True:
        print("c4u6u7.shape:",c4u6u7.shape)
#     c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(4, 4), padding='same')(c4u6)
    c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c4u6)
    c4u6u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c4u6u8)
    if printstate == True:
        print("c4u6u8.shape:",c4u6u8.shape)
#     c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(8, 8), padding='same')(c4u6)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6u9)
    c4u6u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4u6u9)
    if printstate == True:
        print("c4u6u9.shape:",c4u6u9.shape)
    #-----------------------------
    c4u6 = Dropout(dropout)(c4u6)
    c6 = res_blockv2(c4u6,u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    if printstate == True:
        print("c6.shape:",c6.shape)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    if printstate == True:
        print("u7.shape:",u7.shape)
    #---------------------att blk----
    att3 = Attention_block(inputg=u7, inputx=c3, F_int=n_filters * 2)
    if printstate == True:
        print("att3=u7+c3:",att3.shape)
    #---------------------------------
    c3u7 = concatenate([att3, u7,c4u6u7])
    if printstate == True:
        print("attc3-c3u7.shape:",c3u7.shape)
    #-----------------------------
    c3u7u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c3u7)
    if printstate == True:
        print("c3u7u8.shape:",c3u7u8.shape)
#     c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(4, 4), padding='same')(c3u7)
    c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c3u7)
    c3u7u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c3u7u9)
    if printstate == True:
        print("c3u7u9.shape:",c3u7u9.shape)
    #-----------------------------
    c3u7 = Dropout(dropout)(c3u7)
    c7 = res_blockv2(c3u7,u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    if printstate == True:
        print("c7.shape:",c7.shape)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    if printstate == True:
        print("u8.shape:",u8.shape) 
    #---------------------att blk----
    att2 = Attention_block(inputg=u8, inputx=c2, F_int=n_filters * 1)
    if printstate == True:
        print("att2=u8+c2:",att2.shape)
    #---------------------------------
    #-----------------------------
#     c2u8 = concatenate([c2, u8])
    c2u8 = concatenate([att2, u8,c4u6u8,c3u7u8])
    if printstate == True:
        print("attc2-c2u8.shape:",c2u8.shape)
    #-----------------------------
    c2u8u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c2u8)
    if printstate == True:
        print("c2u8u9.shape:",c2u8u9.shape)
    #-----------------------------
    c2u8 = Dropout(dropout)(c2u8)
    c8 = res_blockv2(c2u8,u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    if printstate == True:
        print("c8.shape:",c8.shape)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    if printstate == True:
        print("u9.shape:",u9.shape)
    #---------------------att blk----
    att1 = Attention_block(inputg=u9, inputx=c1, F_int=n_filters /2)
    if printstate == True:
        print("att1=u9+c1:",att2.shape)
    #---------------------------------
#     c1u9 = concatenate([c1, u9], axis=3)
    c1u9 = concatenate([att1, u9,c4u6u9,c3u7u9,c2u8u9], axis=3)
    if printstate == True:
        print("attc2-c2u8.shape:",c1u9.shape)
    c1u9 = Dropout(dropout)(c1u9)
    c9 = res_blockv2(c1u9,u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    if printstate == True:
        print("c9.shape:",c9.shape)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])  # 整个模型的构造
    return model