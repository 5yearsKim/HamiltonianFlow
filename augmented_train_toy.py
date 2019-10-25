import torch
import torch.utils.data
import math
from anode import CNF, AugmentedCNF
import numpy as np
import toy_distribution as toy
import matplotlib.pyplot as plt

import os
from scipy.stats import kde


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dirname = "samples"
if not os.path.exists(dirname):
    os.makedirs(dirname)

batch_size = 100
N_epochs = 60
lr = 0.01

def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2

def compute_loss(x, model):
    # Don't use data parallelize if batch size is small.
    # if x.shape[0] < 200:
    #     model = model.module

    z, sumldj = model(x)  # run model forward
    logpz = standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True).view(-1)  # logp(z)
    logpx = logpz + sumldj

    # loss = -torch.sum(logpx)
    logpx_per_dim = torch.sum(logpx) / x.nelement()  # averaged over batches
    bits_per_dim = -(logpx_per_dim - np.log(256)) / np.log(2)

    return bits_per_dim


def add_augment(x, mode="zero", reverse=False):
    B, W = x.size()
    if reverse:
        x1, x2 = x.view(B,-1,2).chunk(2, dim=2)
        return x1.view(B, -1), x2.view(B, -1)
    else:
        x = x.view(1,-1)
        if mode == "copy":
            aug = x
        elif mode == "zero":
            aug = torch.zeros(x.size())
        return torch.cat([x,aug]).transpose(0,1).contiguous().view(B,-1)


dataset = toy.moon2(10240)
dataset = torch.tensor(dataset).type(torch.float)
trainloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)

model = CNF(4, N_layers=4, Nt=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

total_iter = 0
running_loss = 0.
record_pnt = 100
nbins = 1024

for _ in range(N_epochs):
    for data in trainloader:
        model.train()
        optimizer.zero_grad()

        data = add_augment(data, mode="copy").to(device)
        loss = compute_loss(data, model)
        running_loss += float(loss)

        loss.backward()
        optimizer.step()
        total_iter += 1
        if total_iter % record_pnt ==0:
            mean_loss = running_loss/float(record_pnt)
            print(mean_loss)
            running_loss = 0.

            model.eval()
            with torch.no_grad():
                samples, _ = add_augment(model.sample(100), reverse=True)
                samples = samples.to("cpu").numpy()
                plt.scatter(samples[:,0], samples[:,1], s = 5)

                x = data.detach().to(device)
                y, _ = model(x)
                x, _ = add_augment(x, reverse=True)
                y, _ = add_augment(y, reverse=True)
                x, y = x.to("cpu").numpy(), y.to("cpu").numpy()
                plt.scatter(y[:,0], y[:,1], s = 5)
                plt.scatter(x[:,0],x[:,1], s = 5)
                # x = samples[:,0]
                # y = samples[:,1]
                # k = kde.gaussian_kde([x,y])
                # xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
                # zi = k(np.vstack([xi.flatten(), yi.flatten()]))
                # plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.Greens_r)


                fig_filename = os.path.join(dirname, "moon"+ str(total_iter))
                plt.savefig(fig_filename)
                plt.clf()