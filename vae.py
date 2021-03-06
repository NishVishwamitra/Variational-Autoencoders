# Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).
# A pytorch implementation

from io import open
import glob
import os
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
from torchvision import transforms, utils
import torch
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
from torchvision import models
import sys
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision
import sys

# mnist dataset
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/scratch1/nvishwa', train = True, download = True,
  transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    #torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
    ])),  
  batch_size = 4, shuffle = True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/scratch1/nvishwa', train = False, download = True,
  transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    #torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
  ])),
  batch_size = 1, shuffle = False)

# Conv Autoencoder Model
class Autoencoder(nn.Module):
  def __init__(self, img_embedding_dim):
    super(Autoencoder, self).__init__()
  
    self.img_embedding_dim = img_embedding_dim
    
    # image encoder and decoder ***********************************
    self.img_encoder = nn.Sequential(
      nn.Conv2d(1, 16, kernel_size = 6, stride = 2),
      nn.ReLU(True),
      nn.Conv2d(16, 32, kernel_size = 6, stride = 2),
      nn.ReLU(True),
      nn.Conv2d(32, 64,kernel_size = 4, stride = 2),
      nn.ReLU(True))
      
    #self.img_reduce_dim = nn.Linear(512 * 2 * 2, self.img_embedding_dim)

    # to compute img mean and var
    self.img_lin11 = nn.Linear(64, self.img_embedding_dim)  
    self.img_lin12 = nn.Linear(64, self.img_embedding_dim) 

    self.img_change_dim = nn.Linear(self.img_embedding_dim, 64 * 1 * 1)

    self.img_decoder = nn.Sequential(
      nn.ConvTranspose2d(64, 32, kernel_size = 4, stride = 2),
      nn.ReLU(True),
      nn.ConvTranspose2d(32, 16, kernel_size = 6, stride = 2),
      nn.ReLU(True),
      nn.ConvTranspose2d(16, 1, kernel_size = 6, stride = 2),
      nn.Sigmoid())

  def reparameterize(self, mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = Variable(std.data.new(std.size()).normal_())
    return eps.mul(std).add_(mu)

  def forward(self, x):

    # encode 
    x_e = self.img_encoder(x)
    x_e = x_e.view(x.size(0), -1)
    mu = self.img_lin11(x_e)
    logvar = self.img_lin12(x_e)

    z = self.reparameterize(mu, logvar)

    # decode
    x_d = self.img_change_dim(z)
    x_d = x_d.view(x.size(0), 64, 1, 1)
    x_d = self.img_decoder(x_d)

    return x_d, z, mu, logvar

# Train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 10
annealing_fac = 1. / num_epochs

model = Autoencoder(100).to(device)

recon_x_fun = nn.BCELoss()

def loss_fun(recon_x, x, mu, logvar, epoch, annealing_fac):
  w_kl = epoch * annealing_fac
  BCE = recon_x_fun(recon_x, x)

  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= (x.size(0) ** 5 )

  return BCE + w_kl * KLD, BCE, KLD

optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)

def train(model):
  model.train()
  loss_avg = []
  BCE_avg = []
  KLD_avg = []
  for data in train_loader:
    x, y = data
    x, y = x.to(device), y.to(device, dtype = torch.float32)
    # ===================forward=====================
    x_d, z, mu, logvar = model(x)

    loss, BCE, KLD = loss_fun(x_d, x, mu, logvar, epoch, annealing_fac)
    # ===================backward====================
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_avg.append(loss.item())
    BCE_avg.append(BCE.item())
    KLD_avg.append(KLD.item())
  # ===================log========================
  return sum(loss_avg) / len(loss_avg), sum(BCE_avg) / len(BCE_avg), sum(KLD_avg) / len(KLD_avg)

for epoch in range(num_epochs):
  tr_loss, BCE, KLD = train(model)
  print('epoch [{}/{}], train loss:{:.4f}, BCE loss:{:.4f}, KLD loss:{:.4f}'.format(epoch + 1, num_epochs, tr_loss, BCE, KLD))
torch.save(model, 'vae.pt')
