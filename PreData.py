# -*- coding: utf-8 -*-
# Time : 2024/5/31 16:32
# Author :Liccc
# Email : liccc2332@gmail.com
# File : PreData.py
# Project : KKhyperspectral_5
# back to coding.Keep learning.
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.model_selection import train_test_split
import tifffile as tiff
import numpy as np
import os,cv2

'''制作数据集标签 剔除异常值'''
def ReadData():
    index = 20
    path = r'D:\lc\KKModel\KKhyperspectral_5\0103s\data/'
    save = r'D:\lc\KKModel\KKhyperspectral_5\0103s\label/'
    imgList = os.listdir(path)
    for item in imgList:
        print(item)
        data = tiff.imread(path + '/' + item)
        res = data[:, :, index]
        minRes = np.min(res)
        resIndex = np.unique(res)
        resLimit = resIndex[int(len(resIndex) * 0.95)]
        res[res > resLimit] = minRes
        res = (res - np.min(res)) / (np.max(res) - np.min(res))
        res = np.array(res * 255, dtype=np.uint8)
        res[res != 0] = 255
        cv2.imwrite(save + '/' + item[:-3] + 'png', res)

'''绘制波段曲线图 剔除异常值'''
def ShowData(path):
    data = tiff.imread(path)
    for i in range(data.shape[2]):
        res = data[:,:,i]
        resIndex = np.unique(res)
        minRes = np.min(res)
        resLimit = resIndex[int(len(resIndex) * 0.95)]
        res[res > resLimit] = 0
        index = np.unique(res)
        plt.plot(index)
    plt.show()


'''提取连通域中的均值区域显示'''
def ComponentsData(path1):
    img = cv2.imread(path1,0)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    color_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    sorted_indices = np.argsort(stats[:, cv2.CC_STAT_AREA])[::-1]
    for i, index in enumerate(sorted_indices[1:10], start=1):
        color = list(np.random.random(size=3) * 256)
        color_image[labels == index] = color

    # 显示结果
    cv2.imshow('Segmented Image', color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''制作数据集 划分训练集和测试集 训练集用于数据生成 测试集直接拿去做预测'''
def LoadData(dataPath,labelPath):
    resDatas = []
    resLables = []
    imgList = os.listdir(dataPath)
    for item in imgList:
        print(item)
        tifData = tiff.imread(dataPath+'/'+item)
        img = cv2.imread(labelPath+'/'+item[:-3]+'png',0)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
        sorted_indices = np.argsort(stats[:, cv2.CC_STAT_AREA])[::-1]
        for i, index in enumerate(sorted_indices[1:10], start=1):
            resIndex = np.where(labels == index)
            resTifData = tifData[resIndex[0],resIndex[1]]
            resDatas.append(np.mean(resTifData,axis=0))
            if 'low' in item:
                resLables.append(0)
            else:
                resLables.append(1)
    #划分数据集 测试集保持一致 不进行gan 训练
    X_train, X_test, y_train, y_test = train_test_split(resDatas, resLables, test_size=0.2, stratify=resLables)
    test_indices = np.arange(len(resDatas))[np.isin(resDatas, X_test).all(axis=1)]
    np.save('data/dataTrain.npy',np.array(X_train))
    np.save('data/labelTrain.npy',np.array(y_train))
    np.save('data/dataTest.npy', np.array(X_test))
    np.save('data/labelTest.npy', np.array(y_test))



if __name__ == '__main__':
    # ReadData()#制作数据集label标签
    path1 = r'D:\lc\KKModel\KKhyperspectral_5\ACGan\0103s\data/'
    path2 = r'D:\lc\KKModel\KKhyperspectral_5\ACGan\0103s\label/'
    # ShowData(path)
    LoadData(path1,path2) #制作数据集 分测试集和训练集  测试集不参与训练
    print()