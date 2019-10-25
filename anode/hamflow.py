import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from anode import odesolver_adjoint as odesolver


class ODEBlock(nn.Module):
    def __init__(self, odefunc, Nt):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.options = {}
        self.options.update({'Nt':Nt})
        self.options.update({'method':'RK4'})
        print(self.options)

    def forward(self, x, reverse = False):
        out = odesolver(self.odefunc, x, self.options, reverse = reverse)
        return out

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

# class DNNBlock(nn.Module):
#     def __init__(self, io_dim, hidden_dim, n_Hlayer):
#         assert( io_dim % 2 == 0)
#         super(DNNBlock, self).__init__()
#         self.nfe = 0
#         chain = []
#         chain.append(nn.Linear(io_dim + 1, hidden_dim))
#         assert(n_Hlayer >= 1)
#         for i in range(n_Hlayer):
#             chain.append(nn.Softplus())
#             chain.append(nn.Linear(hidden_dim, hidden_dim))
#         chain.append(nn.Softplus())
#         chain.append(nn.Linear(hidden_dim, io_dim))
#         self.chain = nn.Sequential(*chain)
#
#
#     def forward(self, t, x):
#         self.nfe += 1
#         tt = torch.ones_like(x[:, :1]) * t
#         ttx = torch.cat([tt,x], 1)
#         out = self.chain(ttx)
#         return out



class DNNBlock(nn.Module):
    def __init__(self, io_dim, hidden_dim, n_Hlayer):
        assert(io_dim % 2 == 0)
        super(DNNBlock, self).__init__()
        self.nfe = 0
        chain1 = []
        chain2 = []
        chain1.append(nn.Linear(io_dim//2 + 1, hidden_dim))
        chain2.append(nn.Linear(io_dim//2 + 1, hidden_dim))
        assert(n_Hlayer >= 1)
        for i in range(n_Hlayer):
            chain1.append(nn.Softplus())
            chain2.append(nn.Softplus())
            chain1.append(nn.Linear(hidden_dim, hidden_dim))
            chain2.append(nn.Linear(hidden_dim, hidden_dim))
        chain1.append(nn.Softplus())
        chain2.append(nn.Softplus())
        chain1.append(nn.Linear(hidden_dim, io_dim//2))
        chain2.append(nn.Linear(hidden_dim, io_dim//2))
        self.chain1 = nn.Sequential(*chain1)
        self.chain2 = nn.Sequential(*chain2)


    def forward(self, t, x):
        self.nfe += 1
        B, W = x.size()
        x = x.reshape((B,W//2,2))
        x1, x2 = x[:, :, 0], x[:, :, 1]
        tt1, tt2 = torch.ones_like(x1[:, :1]) * t, torch.ones_like(x2[:, :1]) * t
        ttx1, ttx2 = torch.cat([tt1,x1], 1), torch.cat([tt2, x2], 1)
        out1, out2 = self.chain1(ttx1), self.chain2(ttx2)
        out = torch.stack((out2,out1), dim=2)
        return out.reshape((B,W))


# class Scaling(nn.Module):
#     def __init__(self, io_dim):
#         super(Scaling, self).__init__()
#         self.log_s = nn.Parameter(torch.zeros((1, io_dim)), requires_grad = True)
#
#     def forward(self, x, reverse=False):
#         if reverse:
#             x = x * torch.exp(-self.log_s)
#         else:
#             x = x * torch.exp(self.log_s)
#         return x
#
#     def get_log_det_J(self):
#         return torch.sum(self.log_s)


class AugmentedCNF(nn.Module):
    def __init__(self, io_dim, aug_dim, N_layers):
        super(AugmentedCNF, self).__init__()
        func = []
        chain = []
        for i in range(N_layers):
            func.append(DNNBlock(io_dim + aug_dim, 20, 2))
            chain.append(ODEBlock(func[i],2))
        self.chain = nn.Sequential(
            *chain
        )
        self.scale = Scaling(io_dim)
        self.io_dim = io_dim
        self.aug_dim = aug_dim

    def forward(self, _input, reverse=False):
        B, W = _input.size()
        _input = torch.cat([_input, torch.zeros(B, self.aug_dim)], dim=1)
        if reverse:
            out = self.z_to_x(_input)
        else:
            out = self.x_to_z(_input)
        return out.split(self.io_dim, dim=1)

    def x_to_z(self, x):
        out = self.chain(x)
        return self.scale(out)

    def z_to_x(self, z):
        out = self.scale(z, reverse=True)
        for i in reversed(range(len(self.chain))):
            out = self.chain[i](out, reverse = True)
        return out

    def get_log_det_J(self):
        return self.scale.get_log_det_J()

    def sample(self, size):
        z = torch.randn(size, self.io_dim)
        out, _ = self.forward(z, reverse=True)
        return out

class Additive(nn.Module):
    def __init__(self, io_dim, Hdim, Hlayers, Nt):
        super(Additive, self).__init__()
        self.blk = ODEBlock(DNNBlock(io_dim, Hdim, Hlayers), Nt=Nt)

    def forward(self, x, reverse=False):
        if reverse:
            return self.blk(x, reverse=True)
        else:
            return self.blk(x), 0.


class Scale(nn.Module):
    def __init__(self, io_dim, Hdim, Hlayers, Nt):
        super(Scale, self).__init__()
        self.blk = ODEBlock(DNNBlock(io_dim, Hdim, Hlayers), Nt=Nt)

    def forward(self, x, reverse=False):
        if reverse:
            sign = x.sign()
            logx = x.abs().log()
            logx = self.blk(logx, reverse=True)
            return logx.exp()*sign

        else:
            sign = x.sign()
            logx = x.abs().log()
            ldj1 = -logx.sum(dim=1)
            logx = self.blk(logx)
            ldj2 = logx.sum(dim=1)
            return logx.exp()*sign, ldj1 + ldj2



class CNF(nn.Module):
    def __init__(self, io_dim, N_layers, Hdim=20, Hlayers=3, Nt=1):
        super(CNF, self).__init__()
        chainlist = []
        for i in range(N_layers):
            chainlist.append(Additive(io_dim, Hdim=Hdim, Hlayers=Hlayers, Nt=Nt))
            chainlist.append(Scale(io_dim, Hdim=Hdim, Hlayers=Hlayers, Nt=Nt))
        chainlist.append(Additive(io_dim, Hdim=Hdim, Hlayers=Hlayers, Nt=Nt))
        self.chain = nn.ModuleList(chainlist)
        self.io_dim = io_dim

    def forward(self, _input, reverse=False):
        if reverse:
            return self.z_to_x(_input)
        else:
            return self.x_to_z(_input)

    def x_to_z(self, x):
        out = x
        ldjsum = 0.
        for i in range(len(self.chain)):
            out, ldj = self.chain[i](out)
            ldjsum += ldj
        return out, ldjsum

    def z_to_x(self, z):
        out = z
        for i in reversed(range(len(self.chain))):
            out = self.chain[i](out, reverse=True)
        return out

    # def get_log_det_J(self):
    #     return self.scale.get_log_det_J()

    def sample(self, size):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        z = torch.randn(size, self.io_dim).to(device)
        out = self.forward(z, reverse=True)
        return out



if __name__ == '__main__':

    x = torch.randn(4,2)
    model = CNF(2,10)
    y, ldj = model(x)
    z = model(y, reverse=True)
    print(x)
    print(y)
    print(x-z)