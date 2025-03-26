import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.model_selection import train_test_split
import tifffile as tiff
import numpy as np
import os,cv2



def ReadData(path,save):
    index = 20
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