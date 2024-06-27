# -*- coding: utf-8 -*-
# Time : 2024/5/31 10:31
# Author :Liccc
# Email : liccc2332@gmail.com
# File : GanMain.py
# Project : KKhyperspectral_5
# back to coding.Keep learning.
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from scipy.io import loadmat
import numpy as np
import torch
from NetWork import HyperX,Generator,Discriminator,Classifier,calc_gradient_penalty,reset_grad
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

def Train():
    epoch = 10000
    img = loadmat('PaviaU.mat')['paviaU']
    gt = loadmat('PaviaU_gt.mat')['paviaU_gt']
    mb_size = 512  # Batch size
    z_dim = 30  # Noise dimension
    X_dim = img.shape[-1]  # Number of bands
    h_dim = 512  # Hidden layer size
    d_step = 5  # Number of discriminator training steps for each generator training step
    lr = 5e-5  # Learning rate
    c_weight = 0.2  # Auxiliary classifier weight
    flip_percentage = 0.0  # Proportion of label flipping
    mixup_alpha = 0.1  # Mixup
    semi_supervised = True  # semi-supervision (set to True to include unlabeled samples)
    mask = np.random.randint(0, 100, gt.shape) < 5
    train_gt = np.copy(gt)
    train_gt[np.nonzero(~mask)] = 0
    test_gt = np.copy(gt)
    test_gt[train_gt > 0] = 0
    data_loader = torch.utils.data.DataLoader(
        HyperX(img, train_gt if semi_supervised else gt), batch_size=mb_size, shuffle=True)
    c_dim = data_loader.dataset.n_classes
    opt = [z_dim, c_dim, h_dim, X_dim]
    class_weights = torch.ones((c_dim))
    class_weights[0] = 0.
    class_weights = class_weights.cuda()

    G = Generator(opt).cuda()
    D = Discriminator(opt).cuda()
    C = Classifier(opt).cuda()
    # Use RMSProp optimizer
    G_solver = optim.RMSprop(G.parameters(), lr=lr)
    D_solver = optim.RMSprop(D.parameters(), lr=lr)
    C_solver = optim.RMSprop(C.parameters(), lr=lr)

    for it in tqdm(range(epoch)):
        for p in D.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
        for p in C.parameters():
            p.requires_grad = True

        # D is trained d_step times for each iteration
        for _, (X, y), (X_, y_) in zip(range(d_step), data_loader, data_loader):
            D.zero_grad()

            # Sample random noise
            z = torch.randn(y.size(0), z_dim).squeeze()
            X, y = X.float(), y.float()
            X_, y_ = X_.float(), y_.float()
            # Mixup
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            X = lam * X + (1. - lam) * X_
            X, y, z = X.cuda(), y.cuda(), z.cuda()
            y_ = y_.cuda()
            D_real = D(X).mean()
            fake_samples = G(z, y)
            D_fake = D(fake_samples).mean()
            gradient_penalty = calc_gradient_penalty(D, X.data, fake_samples.data)
            # Compute loss and backpropagate
            D_loss = D_fake - D_real + gradient_penalty
            flip = np.random.random() < flip_percentage
            if flip:
                # Flip real and fake
                gradient_penalty = calc_gradient_penalty(D, fake_samples.data, X.data)
                D_loss = D_real - D_fake + gradient_penalty
            D_loss.backward()
            D_solver.step()

            ######################
            #  Update C network  #
            ######################
            C.zero_grad()
            # Get class values
            _, classes = torch.max(y, dim=1)
            _, classes_ = torch.max(y_, dim=1)
            # Get predictions from C
            if flip:
                fake_samples = G(z, y)
                pred = C(fake_samples)
                # Compute loss and backpropagate
                C_loss = F.cross_entropy(pred, classes, weight=class_weights)
            else:
                pred = F.log_softmax(C(X))
                C_loss = lam * F.nll_loss(pred, classes) + (1. - lam) * F.nll_loss(pred, classes_)
            C_loss.backward()
            C_solver.step()

        ############################
        # (2) Update G network
        ###########################
        for p in D.parameters():
            p.requires_grad = False  # to avoid computation
        for p in C.parameters():
            p.requires_grad = False
        reset_grad(C, G, D)

        # Sample random noise
        z = torch.randn(y.size(0), z_dim).squeeze()
        z = z.cuda()
        G_sample = G(z, y)
        D_fake = D(G_sample)
        pred = C(G_sample)
        C_loss = F.cross_entropy(pred, classes, weight=class_weights)
        # Fool the discriminator (WGAN)
        G_loss = -torch.mean(D_fake)
        # Include the auxialiary classifier loss (AC-GAN)
        loss = G_loss + c_weight * C_loss
        # Backpropagate
        loss.backward()
        G_solver.step()

        # Print and plot every now and then
        if it % 1000 == 0:
            with torch.no_grad():
                print('Iter-{}; D_loss: {}; G_loss: {}; C_loss: {}'.format(it,
                                                                           D_loss.data.cpu().numpy(),
                                                                           G_loss.data.cpu().numpy(),
                                                                           C_loss.data.cpu().numpy()))
                z = torch.randn(mb_size, z_dim).squeeze().cuda()
                c = np.zeros(shape=[mb_size, c_dim], dtype='float32')
                idx = np.random.randint(1, data_loader.dataset.n_classes)
                c[:, idx] = 1.
                c = torch.from_numpy(c).squeeze().cuda()
                samples = G(z, c).data.cpu().numpy()[:16]
                pred = G(z, c)
                torch.save(G.state_dict(), 'bestData.pth')

def TrainV1(epoch,modelPath):
    img = np.load('data/dataTrain.npy')[:1650]
    img = img.reshape([75, 22, img.shape[1]])
    gt = np.load('data/labelTrain.npy')[:1650]
    gt = gt.reshape([75,22])
    mb_size = 24  # Batch size
    z_dim = 30  # Noise dimension
    X_dim = img.shape[-1]  # Number of bands
    h_dim = 512  # Hidden layer size
    d_step = 5  # Number of discriminator training steps for each generator training step
    lr = 5e-5  # Learning rate
    c_weight = 0.2  # Auxiliary classifier weight
    flip_percentage = 0.0  # Proportion of label flipping
    mixup_alpha = 0.1  # Mixup
    semi_supervised = True  # semi-supervision (set to True to include unlabeled samples)
    mask = np.random.randint(0, 100, gt.shape) < 5
    train_gt = np.copy(gt)
    train_gt[np.nonzero(~mask)] = 0
    test_gt = np.copy(gt)
    test_gt[train_gt > 0] = 0
    data_loader = torch.utils.data.DataLoader(
        HyperX(img, train_gt if semi_supervised else gt), batch_size=mb_size, shuffle=True)
    c_dim = data_loader.dataset.n_classes
    opt = [z_dim, c_dim, h_dim, X_dim]
    class_weights = torch.ones((c_dim))
    class_weights[0] = 0.
    class_weights = class_weights.cuda()

    G = Generator(opt).cuda()
    D = Discriminator(opt).cuda()
    C = Classifier(opt).cuda()
    # Use RMSProp optimizer
    G_solver = optim.RMSprop(G.parameters(), lr=lr)
    D_solver = optim.RMSprop(D.parameters(), lr=lr)
    C_solver = optim.RMSprop(C.parameters(), lr=lr)

    for it in tqdm(range(epoch)):
        for p in D.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
        for p in C.parameters():
            p.requires_grad = True

        # D is trained d_step times for each iteration
        for _, (X, y), (X_, y_) in zip(range(d_step), data_loader, data_loader):
            D.zero_grad()
            # Sample random noise
            z = torch.randn(y.size(0), z_dim).squeeze()
            X, y = X.float(), y.float()
            X_, y_ = X_.float(), y_.float()
            # Mixup
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            X = lam * X + (1. - lam) * X_
            X, y, z = X.cuda(), y.cuda(), z.cuda()
            y_ = y_.cuda()
            D_real = D(X).mean()
            fake_samples = G(z, y)
            D_fake = D(fake_samples).mean()
            gradient_penalty = calc_gradient_penalty(D, X.data, fake_samples.data)
            # Compute loss and backpropagate
            D_loss = D_fake - D_real + gradient_penalty
            flip = np.random.random() < flip_percentage
            if flip:
                # Flip real and fake
                gradient_penalty = calc_gradient_penalty(D, fake_samples.data, X.data)
                D_loss = D_real - D_fake + gradient_penalty
            D_loss.backward()
            D_solver.step()

            ######################
            #  Update C network  #
            ######################
            C.zero_grad()
            # Get class values
            _, classes = torch.max(y, dim=1)
            _, classes_ = torch.max(y_, dim=1)
            # Get predictions from C
            if flip:
                fake_samples = G(z, y)
                pred = C(fake_samples)
                # Compute loss and backpropagate
                C_loss = F.cross_entropy(pred, classes, weight=class_weights)
            else:
                pred = F.log_softmax(C(X))
                C_loss = lam * F.nll_loss(pred, classes) + (1. - lam) * F.nll_loss(pred, classes_)
            C_loss.backward()
            C_solver.step()
        for p in D.parameters():
            p.requires_grad = False  # to avoid computation
        for p in C.parameters():
            p.requires_grad = False
        reset_grad(C, G, D)

        z = torch.randn(y.size(0), z_dim).squeeze()
        z = z.cuda()
        G_sample = G(z, y)
        D_fake = D(G_sample)
        pred = C(G_sample)
        C_loss = F.cross_entropy(pred, classes, weight=class_weights)
        # Fool the discriminator (WGAN)
        G_loss = -torch.mean(D_fake)
        # Include the auxialiary classifier loss (AC-GAN)
        loss = G_loss + c_weight * C_loss
        # Backpropagate
        loss.backward()
        G_solver.step()

        # Print and plot every now and then
        if it % 500 == 0:
            with torch.no_grad():
                print('Iter-{}; D_loss: {}; G_loss: {}; C_loss: {}'.format(it,
                                                                           D_loss.data.cpu().numpy(),
                                                                           G_loss.data.cpu().numpy(),
                                                                           C_loss.data.cpu().numpy()))
                z = torch.randn(mb_size, z_dim).squeeze().cuda()
                c = np.zeros(shape=[mb_size, c_dim], dtype='float32')
                idx = np.random.randint(1, data_loader.dataset.n_classes)
                c[:, idx] = 1.
                c = torch.from_numpy(c).squeeze().cuda()
                # samples = G(z, c).data.cpu().numpy()[:16]
                # pred = G(z, c)
                torch.save(G.state_dict(), modelPath)


'''生成器代码'''
def GenerateData(path):
    num = 75*50
    resData = []
    resLabel = []
    img = np.load('data/dataTrain.npy')[:1650]
    img = img.reshape([75, 22, img.shape[1]])
    gt = np.load('data/labelTrain.npy')[:1650]
    gt = gt.reshape([75, 22])
    mb_size = 24  # Batch size
    z_dim = 30  # Noise dimension
    X_dim = img.shape[-1]  # Number of bands
    h_dim = 512  # Hidden layer size
    c_dim = np.max(gt)+1
    opt = [z_dim, c_dim, h_dim, X_dim]
    net = Generator(opt).cuda()
    net.load_state_dict(torch.load(path))
    net.eval()
    z = torch.randn(mb_size, z_dim).squeeze().cuda()
    for i in range(num):
        for idx in range(c_dim):
            c = np.zeros(shape=[mb_size, c_dim], dtype='float32')
            c[:, idx] = 1.
            c = torch.from_numpy(c).squeeze().cuda()
            samples = net(z, c).data.cpu().numpy()
            resData.append(np.mean(samples,axis=0))
            resLabel.append(idx)
    img1 = img.reshape([img.shape[0] * img.shape[1], img.shape[2]])
    gt = gt.reshape([gt.shape[0]* gt.shape[1]])
    resTFData = np.vstack([img1,np.array(resData)])
    gt = np.hstack([gt,np.array(resLabel)])
    resTFData = resTFData.reshape([img.shape[0],int(resTFData.shape[0]/img.shape[0]),img.shape[2]])
    gt = gt.reshape([resTFData.shape[0],resTFData.shape[1]])
    return resTFData,gt




if __name__ == '__main__':
    #训练不同轮数的模型
    # epochs = [100,500,1000,2000,5000,8000,10000]
    # TrainV1(100, 'test')  # 训练
    # for epoch in epochs:
    #     name = 'model/model_'+str(epoch)+'.pth'
    #     TrainV1(epoch,name)   #训练


    #生成数据
    path = r'bestData.pth'
    data,label = GenerateData(path)  #生成器生成新数据
    np.save('data/dataV1.npy',data)
    np.save('data/labelV1.npy',label)
    # data = np.load('data/dataV1.npy')
    print()
