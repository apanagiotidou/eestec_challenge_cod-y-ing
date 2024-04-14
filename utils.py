import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2
import random
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def getName(filePath):
    return filePath.split('\\')[-1]


def importDataInfo(path):
    columns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    data = pd.read_csv(os.path.join(path, 'driving_log.csv'), names=columns)

    print(data.head())
    print(data['Center'][0])
    print(getName(data['Center'][1]))
    data['Center'] = data['Center'].apply(getName)

    print("Total Images: ", data.shape[0])

    return data

def balanceData(data, display=True):
    nBins = 31
    samplesPerBin = 1200
    hist, bins = np.histogram(data['Steering'], nBins)
    #print(bins)
    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.bar(center, hist, width=0.06)
        plt.plot((-1,1), (samplesPerBin, samplesPerBin))
        plt.show()

    removeIndexList = []
    for j in range(nBins):
        binDataList = []
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j+1] :
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeIndexList.extend(binDataList)

    print("Removed Images: ", len(removeIndexList))
    data.drop(data.index[removeIndexList], inplace=True)
    print("Remaining Images: ", len(data))


    if display:
        hist, _ = np.histogram(data['Steering'], nBins)
        plt.bar(center, hist, width=0.06)
        plt.plot((-1,1), (samplesPerBin, samplesPerBin))
        plt.show()

    return data

def loadData(path, data):
    imagePath= []
    steering = []

    for i in range(len(data)):
        indexedData = data.iloc[i]
        #print(indexedData)
        imagePath.append(os.path.join(path,'IMG',indexedData[0]))
        #print(os.path.join(path,'IMG',indexedData[0]))
        steering.append(float(indexedData[3]))
    imagePath = np.asarray(imagePath)
    steering = np.asarray(steering)
    return imagePath, steering

def shadow(image):
    top_y = 320 * np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320 * np.random.uniform()
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    shadow_mask = 0 * image_hls[:, :, 1]
    # print shadow_mask
    X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]
    shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >= 0)] = 1
    # print shadow_mask
    # random_bright = .25+.7*np.random.uniform()
    # if np.random.randint(2)==1:
    random_bright = .5
    cond1 = shadow_mask == 1
    cond0 = shadow_mask == 0
    # print shadow_mask
    if np.random.randint(2) == 1:
        image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1] * random_bright
    # print '111'
    else:
        image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0] * random_bright
    # print '111'
    image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)
    return image

def augmentImage(imgPath, steering):
    img = mpimg.imread(imgPath)
    # PAN
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={'x':(-0.1, 0.1), 'y':(-0.1, 0.1)})
        img = pan.augment_image(img)
    # ZOOM
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
    # BRIGHTNESS
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.4, 1.2))
        img = brightness.augment_image(img)
    # FLIP
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering

    # RANDOM BLUR   NEW
    if np.random.rand() < 0.5:
        size = 1 + int(np.random.rand() * 7)
        if size % 2 == 0:
            size = size + 1
        img = cv2.GaussianBlur(img, (size, size), 0)

    # SHADOW  NEW
    if np.random.rand() < 0.5:
        img = shadow(img)


    return img, steering

def preProcessing(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img

def batchGen(imagesPath, steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []
        for i in range(batchSize):
            index = random.randint(0, len(imagesPath)-1)
            if trainFlag:
                img, steering = augmentImage(imagesPath[index], steeringList[index])
            else:
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index]
            img = preProcessing(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield (np.asarray(imgBatch), np.asarray(steeringBatch))

def createModel():
    model = Sequential()
    model.add(Convolution2D(24, (5,5), (2,2), input_shape=(66,200,3), activation='elu'))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.compile(Adam(lr=0.0001), loss='mse')

    return model

#imgRe, steering = augmentImage('test.jpg', 0)
#plt.imshow(imgRe)
#plt.show()