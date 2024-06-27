# -*- coding: utf-8 -*-
# Time : 2024/6/4 19:56
# Author :Liccc
# Email : liccc2332@gmail.com
# File : ShowData.py
# Project : ACGan
# back to coding.Keep learning.
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
from NetWork import HyperX,Generator,Discriminator,Classifier,calc_gradient_penalty,reset_grad
import torch
import os
'''效果图代码
1.不同训练轮数中的数据产生和真实标签比较 真实标签数据    
2.不同比例假数据训练结果比较

'''
def PreData(path):
    img = np.load('data/dataTrain.npy')[:1650]
    img = img.reshape([75, 22, img.shape[1]])
    gt = np.load('data/labelTrain.npy')[:1650]
    gt = gt.reshape([75, 22])
    mb_size = 24  # Batch size
    z_dim = 30  # Noise dimension
    X_dim = img.shape[-1]  # Number of bands
    h_dim = 512  # Hidden layer size
    c_dim = np.max(gt) + 1
    opt = [z_dim, c_dim, h_dim, X_dim]
    net = Generator(opt).cuda()
    net.load_state_dict(torch.load(path))
    net.eval()
    z = torch.randn(mb_size, z_dim).squeeze().cuda()
    ress = []
    for idx in range(c_dim):
        c = np.zeros(shape=[mb_size, c_dim], dtype='float32')
        c[:, idx] = 1.
        c = torch.from_numpy(c).squeeze().cuda()
        samples = net(z, c).data.cpu().numpy() #生成器生成的数据
        samples = np.mean(samples, axis=0)
        samples = (samples-np.min(samples))/(np.max(samples)-np.min(samples))
        ress.append(samples)
        # ress.append(samples[int(mb_size/2)])
    return ress

'''不同训练轮数的曲线图'''
def ShowEpochData():
    res0 = []
    res1 = []
    modelPath = r'model'
    items = ['model_100.pth','model_500.pth','model_1000.pth','model_2000.pth','model_5000.pth','model_8000.pth','model_10000.pth']
    x = [400+i*3.22 for i in range(186)]
    for path in items:
        data = PreData(modelPath+'/'+path)
        res0.append(data[0])
        res1.append(data[1])

    res0ture = np.load('data/dataTrain.npy')[3]
    res0ture = (res0ture-np.min(res0ture))/(np.max(res0ture)-np.min(res0ture))
    res1ture = np.load('data/dataTrain.npy')[0]
    res1ture = (res1ture - np.min(res1ture)) / (np.max(res1ture) - np.min(res1ture))

    fig, axs = plt.subplots(4, 2, figsize=(20, 20),gridspec_kw={'hspace': 0.8, 'wspace': 0.2})
    # fig.suptitle('不同轮次生成器数据对比图',fontsize=25)
    axs[0, 0].plot(x,res1[0], 'r-',label='class:high')
    axs[0, 0].plot(x, res0[0], 'b-', label='class:low')
    axs[0, 0].set_ylabel('Normalized Reflectance')
    axs[0, 0].set_title('Generated spectrum Epoch:100')
    axs[0, 0].set_xlabel('Wavelength (nm)')
    # axs[0, 0].legend()

    axs[0, 1].plot(x,res1[1], 'r-', label='class:high')
    axs[0, 1].plot(x, res0[1], 'b-', label='class:low')
    axs[0, 1].set_ylabel('Normalized Reflectance')
    axs[0, 1].set_title('Generated spectrum Epoch:500')
    axs[0, 1].set_xlabel('Wavelength (nm)')
    # axs[0, 1].legend()

    axs[1, 0].plot(x,res1[2], 'r-', label='class:high')
    axs[1, 0].plot(x, res0[2], 'b-', label='class:low')
    axs[1, 0].set_title('Generated spectrum Epoch:1000')
    axs[1, 0].set_ylabel('Normalized Reflectance')
    axs[1, 0].set_xlabel('Wavelength (nm)')
    # axs[1, 0].legend()

    axs[1, 1].plot(x,res1[3], 'r-', label='class:high')
    axs[1, 1].plot(x, res0[3], 'b-', label='class:low')
    axs[1, 1].set_ylabel('Normalized Reflectance')
    axs[1, 1].set_title('Generated spectrum Epoch:2000')
    axs[1, 1].set_xlabel('Wavelength (nm)')
    # axs[1, 1].legend()

    axs[2, 0].plot(x,res1[4], 'r-', label='class:high')
    axs[2, 0].plot(x, res0[4], 'b-', label='class:low')
    axs[2, 0].set_ylabel('Normalized Reflectance')
    axs[2, 0].set_title('Generated spectrum Epoch:5000')
    axs[2, 0].set_xlabel('Wavelength (nm)')
    # axs[2, 0].legend()

    axs[2, 1].plot(x,res1[5], 'r-', label='class:high')
    axs[2, 1].plot(x, res0[5], 'b-', label='class:low')
    axs[2, 1].set_title('Generated spectrum Epoch:8000')
    axs[2, 1].set_ylabel('Normalized Reflectance')
    axs[2, 1].set_xlabel('Wavelength (nm)')
    # axs[2, 1].legend()

    axs[3, 0].plot(x,res1[6], 'r-', label='class:high')
    axs[3, 0].plot(x, res0[6], 'b-', label='class:low')
    axs[3, 0].set_title('Generated spectrum Epoch:10000')
    axs[3, 0].set_xlabel('Wavelength (nm)')
    axs[3, 0].set_ylabel('Normalized Reflectance')
    # axs[3, 0].legend()

    axs[3, 1].plot(x,res1ture, 'r-', label='class:high')
    axs[3, 1].plot(x, res0ture, 'b-', label='class:low')
    axs[3, 1].set_title('Real spectrum')
    axs[3, 1].set_ylabel('Normalized Reflectance')
    axs[3, 1].set_xlabel('Wavelength (nm)')
    # axs[3, 1].legend()
    plt.show()





if __name__ == '__main__':
    ShowEpochData()












