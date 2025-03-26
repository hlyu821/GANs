import numpy as np
import os,cv2
import pandas as pd
import tifffile as tiff
from sklearn.model_selection import train_test_split


def LoadData_ml (dataPath, labelPath):
    data_rows = []  # 存储所有数据行
    
    # 获取标签路径下的所有文件列表
    imgList = os.listdir(labelPath)
    
    # 遍历每个文件
    for item in imgList:
        print(f"Processing file: {item}")
        
        # 提取前缀以匹配 TIFF 文件
        prefix = '_'.join(item.split('_')[:2])  # 如 h1_0301
        
        # 查找匹配的 TIFF 文件
        tif_file_name = None
        for tif_item in os.listdir(dataPath):
            if prefix in tif_item and tif_item.endswith('.tif'):
                tif_file_name = tif_item
                break
        
        if tif_file_name is None:
            print(f"No matching TIFF file found for prefix: {prefix}")
            continue
        
        tif_file_path = os.path.join(dataPath, tif_file_name)
        tifData = tiff.imread(tif_file_path)
        
        # 读取标签图像（灰度模式）
        png_file_path = os.path.join(labelPath, item)
        img = cv2.imread(png_file_path, 0)
        
        if img is None:
            print(f"Failed to read image: {png_file_path}")
            continue
        
        # 处理连通组件
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
        sorted_indices = np.argsort(stats[:, cv2.CC_STAT_AREA])[::-1]
        
        # 存储前九个连通组件的均值
        mean_values = []
        
        # 遍历前 9 个最大的连通组件（跳过背景）
        for i, index in enumerate(sorted_indices[1:10], start=1):
            resIndex = np.where(labels == index)
            resTifData = tifData[resIndex[0], resIndex[1]]
            mean_data = np.mean(resTifData, axis=0)
            mean_values.append(mean_data)
        
        # 计算前九个连通组件的均值的均值
        if mean_values:
            overall_mean = np.mean(mean_values, axis=0)
            
            # 构建数据行：[图片名称, 标签, 特征值...]
            label = 0 if 'low' in item else 1
            row = [prefix, label] + overall_mean.tolist()
            data_rows.append(row)
    
    # 转换为DataFrame
    columns = ['image_name', 'label'] + [f'feature_{i}' for i in range(len(data_rows[0])-2)]
    df = pd.DataFrame(data_rows, columns=columns)
    
    # 划分数据集
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    
    # 保存数据
    save_dir = 'data12'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    train_df.to_csv(os.path.join(save_dir, 'dataTrain.csv'), index=False)
    test_df.to_csv(os.path.join(save_dir, 'dataTest.csv'), index=False)

def LoadData_ml_reg(dataPath, labelPath):
    data_rows = []  # 存储所有数据行
    
    # 获取标签路径下的所有文件列表
    imgList = os.listdir(labelPath)
    
    # 遍历每个文件
    for item in imgList:
        print(f"Processing file: {item}")
        
        # 提取前缀以匹配 TIFF 文件
        prefix = '_'.join(item.split('_')[:2])  # 如 h1_0301
        
        # 查找匹配的 TIFF 文件
        tif_file_name = None
        for tif_item in os.listdir(dataPath):
            if prefix in tif_item and tif_item.endswith('.tif'):
                tif_file_name = tif_item
                break
        
        if tif_file_name is None:
            print(f"No matching TIFF file found for prefix: {prefix}")
            continue
        
        tif_file_path = os.path.join(dataPath, tif_file_name)
        tifData = tiff.imread(tif_file_path)
        
        # 读取标签图像（灰度模式）
        png_file_path = os.path.join(labelPath, item)
        img = cv2.imread(png_file_path, 0)
        
        if img is None:
            print(f"Failed to read image: {png_file_path}")
            continue
        
        # 处理连通组件
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
        sorted_indices = np.argsort(stats[:, cv2.CC_STAT_AREA])[::-1]
        
        # 存储前九个连通组件的均值
        mean_values = []
        
        # 遍历前 9 个最大的连通组件（跳过背景）
        for i, index in enumerate(sorted_indices[1:10], start=1):
            resIndex = np.where(labels == index)
            resTifData = tifData[resIndex[0], resIndex[1]]
            mean_data = np.mean(resTifData, axis=0)
            mean_values.append(mean_data)
        
        # 计算前九个连通组件的均值的均值
        if mean_values:
            overall_mean = np.mean(mean_values, axis=0)
            
            # 提取并转换标签
            label_str = item.split('_')[-2] + '.' + item.split('_')[-1].split('.')[0]
            label = float(label_str)
            
            # 构建数据行：[图片名称, 标签, 特征值...]
            row = [prefix, label] + overall_mean.tolist()
            data_rows.append(row)
    
    # 转换为DataFrame
    columns = ['image_name', 'label'] + [f'feature_{i}' for i in range(len(data_rows[0])-2)]
    df = pd.DataFrame(data_rows, columns=columns)
    
    # 划分数据集
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # 保存数据
    save_dir = 'data12'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    train_df.to_csv(os.path.join(save_dir, 'dataTrain_reg.csv'), index=False)
    test_df.to_csv(os.path.join(save_dir, 'dataTest_reg.csv'), index=False)
