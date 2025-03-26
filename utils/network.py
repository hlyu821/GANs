import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import torch
import numpy as np
import torch.nn as nn

def weight_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d)):
        torch.nn.init.kaiming_normal_(m.weight.data)

class HyperX(torch.utils.data.Dataset):
    def __init__(self, data, labels, semi=False):
        super(HyperX, self).__init__()
        # Normalize the data in [0,1]
        # data = (data - data.min()) / (data.max() - data.min())
        self.data = data
        self.labels = labels
        self.n_classes = len(np.unique(labels))
        self.semi = semi
        if semi:
            # Semi-supervision logic
            labeled_indices = np.nonzero(labels > 0)[0]
        else:
            labeled_indices = np.arange(len(labels))
        self.indices = labeled_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 获取单个数据样本
        data = self.data[idx]

        # 获取对应的标签，并确保其为整数
        label_idx = int(self.labels[idx])

        # 检查标签索引是否在有效范围内
        if label_idx < 0 or label_idx >= self.n_classes:
            raise IndexError(f"Label index {label_idx} out of range for {self.n_classes} classes.")

        # 将标签转换为one-hot编码向量， 处理连续数值变量
        label = np.asarray(np.eye(self.n_classes)[label_idx], dtype='int64')

        return torch.from_numpy(data), torch.from_numpy(label)

class Generator(nn.Module):
    def __init__(self,arges):
        super(Generator, self).__init__()
        z_dim,c_dim,h_dim,X_dim = arges[0],arges[1],arges[2],arges[3]
        # LeakyReLU is preferred to keep gradients flowing even for negative activations
        self.generator = torch.nn.Sequential(
            torch.nn.Linear(z_dim + c_dim, h_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(h_dim, h_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(h_dim, h_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(h_dim, X_dim),
            torch.nn.Sigmoid()  # smooth [0,1] outputs
        )
        self.apply(weight_init)

    def forward(self, z, c):
        # Concatenate the noise and condition
        inputs = torch.cat([z, c], 1)
        return self.generator(inputs)



# 鉴别器 discriminator: sample -> -infty -- fake - 0 - real -- +infty
class Discriminator(nn.Module):
    def __init__(self,arges):
        super(Discriminator, self).__init__()
        z_dim, c_dim, h_dim, X_dim = arges[0], arges[1], arges[2], arges[3]
        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(X_dim, h_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(h_dim, h_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(h_dim, h_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(h_dim, 1)
        )

        self.apply(weight_init)

    def forward(self, X):
        return self.discriminator(X)




# Basic fully connected classifier: sample -> class
class Classifier(nn.Module):
    def __init__(self,arges):
        super(Classifier, self).__init__()
        z_dim, c_dim, h_dim, X_dim = arges[0], arges[1], arges[2], arges[3]
        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(X_dim, h_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(h_dim, h_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(h_dim, h_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(h_dim, c_dim)
        )
        self.apply(weight_init)
    def forward(self, X):
        return self.discriminator(X)

def calc_gradient_penalty(netD, real_data, generated_data, penalty_weight=10):
    batch_size = real_data.size()[0]

    alpha = torch.rand(batch_size, 1) if real_data.dim() == 2 else torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.cuda()

    interpolated = alpha * real_data + (1 - alpha) * generated_data
    interpolated.requires_grad_(True)
    interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = netD(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                    grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                                    create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(batch_size, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return penalty_weight * ((gradients_norm - 1) ** 2).mean()


def reset_grad(*nets):
    for net in nets:
        net.zero_grad()