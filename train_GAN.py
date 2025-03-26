import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from utils.network import HyperX, Generator, Discriminator, Classifier, calc_gradient_penalty, reset_grad
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np

def TrainV1_ml(epoch, modelPath):
    # Load data from CSV
    df = pd.read_csv('data12/dataTrain.csv', index_col=0)
    gt, _ = pd.factorize(df.iloc[:, 0].values)
    print(gt)
    img = df.iloc[:, 1:].values

    # Parameters
    mb_size = 32  # Batch size
    z_dim = 30  # Noise dimension
    X_dim = img.shape[1]  # Number of features (bands)
    print(X_dim)
    h_dim = 512  # Hidden layer size
    d_step = 5  # Number of discriminator training steps for each generator training step
    lr = 5e-5  # Learning rate
    c_weight = 0.2  # Auxiliary classifier weight
    flip_percentage = 0.0  # Proportion of label flipping
    mixup_alpha = 0.1  # Mixup

    data_loader = torch.utils.data.DataLoader(
        HyperX(img, gt, semi=False), batch_size=mb_size, shuffle=True)

    print(f'Number of classes: {data_loader.dataset.n_classes}')

    c_dim = data_loader.dataset.n_classes
    opt = [z_dim, c_dim, h_dim, X_dim]
    class_weights = torch.ones((c_dim))
    class_weights[0] = 0.
    class_weights = class_weights.cuda()

    G = Generator(opt).cuda()
    D = Discriminator(opt).cuda()
    C = Classifier(opt).cuda()
    G_solver = optim.RMSprop(G.parameters(), lr=lr)
    D_solver = optim.RMSprop(D.parameters(), lr=lr)
    C_solver = optim.RMSprop(C.parameters(), lr=lr)

    for it in tqdm(range(epoch)):
        for p in D.parameters():  # reset requires_grad
            p.requires_grad = True
        for p in C.parameters():
            p.requires_grad = True

        for _, (X, y), (X_, y_) in zip(range(d_step), data_loader, data_loader):
            D.zero_grad()
            z = torch.randn(y.size(0), z_dim).squeeze()
            X, y = X.float(), y.float()
            X_, y_ = X_.float(), y_.float()
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            X = lam * X + (1. - lam) * X_
            X, y, z = X.cuda(), y.cuda(), z.cuda()
            y_ = y_.cuda()
            D_real = D(X).mean()
            fake_samples = G(z, y)
            D_fake = D(fake_samples).mean()
            gradient_penalty = calc_gradient_penalty(D, X.data, fake_samples.data)
            D_loss = D_fake - D_real + gradient_penalty
            flip = np.random.random() < flip_percentage
            if flip:
                gradient_penalty = calc_gradient_penalty(D, fake_samples.data, X.data)
                D_loss = D_real - D_fake + gradient_penalty
            D_loss.backward()
            D_solver.step()

            C.zero_grad()
            _, classes = torch.max(y, dim=1)
            _, classes_ = torch.max(y_, dim=1)
            if flip:
                fake_samples = G(z, y)
                pred = C(fake_samples)
                C_loss = F.cross_entropy(pred, classes, weight=class_weights)
            else:
                pred = F.log_softmax(C(X))
                C_loss = lam * F.nll_loss(pred, classes) + (1. - lam) * F.nll_loss(pred, classes_)
            C_loss.backward()
            C_solver.step()
        for p in D.parameters():
            p.requires_grad = False
        for p in C.parameters():
            p.requires_grad = False
        reset_grad(C, G, D)

        z = torch.randn(y.size(0), z_dim).squeeze()
        z = z.cuda()
        G_sample = G(z, y)
        D_fake = D(G_sample)
        pred = C(G_sample)
        C_loss = F.cross_entropy(pred, classes, weight=class_weights)
        G_loss = -torch.mean(D_fake)
        loss = G_loss + c_weight * C_loss
        loss.backward()
        G_solver.step()

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
                generated_samples = G(z, c).cpu().numpy()
                print(f"Generated samples stats: min={np.min(generated_samples)}, max={np.max(generated_samples)}, mean={np.mean(generated_samples)}")
                torch.save(G.state_dict(), modelPath)








if __name__ == '__main__':
    #训练不同轮数的模型
     epochs = [500, 1000, 2000, 5000, 8000, 10000, 20000]
    # TrainV1(100, 'test')  # 训练
     for epoch in epochs:
         name = 'model12/model_class_'+str(epoch)+'.pth'
         TrainV1_ml(epoch,name)   #训练



