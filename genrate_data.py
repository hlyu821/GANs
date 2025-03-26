import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from utils.network import  Generator

import torch
import pandas as pd
import numpy as np
import os


def GenerateData(path, data_save_path, labels_save_path):
    num = 4*100  # Number of samples to generate
    resData = []
    resLabel = []
    # Load data from CSV
    df = pd.read_csv('data12/dataTrain.csv', index_col=0)
    labels, _ = pd.factorize(df.iloc[:, 0].values)
    print(labels)
    data = df.iloc[:, 1:].values

    # Parameters
    mb_size = 32  # Batch size
    z_dim = 30  # Noise dimension
    X_dim = data.shape[1]  # Number of features (bands)
    print(X_dim)
    h_dim = 512  # Hidden layer size
    c_dim = np.max(labels) + 1  # Number of classes
    print(c_dim)
    opt = [z_dim, c_dim, h_dim, X_dim]

    net = Generator(opt).cuda()
    net.load_state_dict(torch.load(path))
    net.eval()

    for idx in range(c_dim):
        c = np.zeros(shape=[mb_size, c_dim], dtype='float32')
        c[:, idx] = 1.0
        c = torch.from_numpy(c).squeeze().cuda()

        for _ in range(num // c_dim):  # Generate multiple samples per class
            z = torch.randn(mb_size, z_dim).squeeze().cuda()  # Generate new noise vector
            samples = net(z, c).data.cpu().numpy()
            resData.append(np.mean(samples, axis=0))
            resLabel.append(idx)

    # Convert lists to numpy arrays
    resData = np.array(resData)
    resLabel = np.array(resLabel)

    # Save the data and labels to .npy files
    np.save(data_save_path, resData)
    np.save(labels_save_path, resLabel)

    return resData, resLabel

def save_data_and_labels_to_csv(data, labels, csv_path):
    # Convert data and labels to a DataFrame
    # Flatten the data to make it compatible with DataFrame structure
    num_samples = data.shape[0]
    num_features = data.shape[1]

    # Flatten the data to 2D
    flat_data = data.reshape(num_samples, num_features)

    # Combine labels with the flattened data
    df = pd.DataFrame(flat_data)
    df.insert(0, 'label', labels)

    # Save the DataFrame to a CSV file
    df.to_csv(csv_path, index=False)
    print(f'Data and labels saved to {csv_path}')


# if __name__ == '__main__':
#     path = r'D:\GAN\ACGan\ACGan\model12\model_500.pth'
#     data, labels = GenerateData(path)  # 生成数据
#     save_data_and_labels_to_csv(data, labels, 'generated_data_labels_500.csv')  # 保存为 CSV 文件

if __name__ == '__main__':

    output_dir = './result'

    epochs = [500, 1000, 2000, 5000, 8000, 10000, 20000]

    # 依次读取生成数据并保存
    for epoch in epochs:
        path = 'model12/model_class_'+str(epoch)+'.pth'
        data, labels = GenerateData(path)  # 生成数据
        output_filename = os.path.join(output_dir, f'generated_data)_{epoch}.csv')
        save_data_and_labels_to_csv(data, labels, output_filename)  # 保存为 CSV 文件