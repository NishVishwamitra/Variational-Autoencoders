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
import matplotlib.pyplot as plt
# mnist dataset
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/scratch1/nvishwa', train = False, download = True,
  transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    #torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
  ])),
  batch_size = 1, shuffle = True)

# Conv Autoencoder Model
class Autoencoder(nn.Module):
  def __init__(self, img_embedding_dim):
    super(Autoencoder, self).__init__()
  
    self.img_embedding_dim = img_embedding_dim
    
    # image encoder and decoder ***********************************
    self.img_encoder = nn.Sequential(
      nn.Conv2d(2, 16, kernel_size = 6, stride = 2),
      nn.ReLU(True),
      nn.Conv2d(16, 32, kernel_size = 6, stride = 2),
      nn.ReLU(True),
      nn.Conv2d(32, 64,kernel_size = 4, stride = 2),
      nn.ReLU(True))

    # to compute img mean and var
    self.img_lin11 = nn.Linear(64, self.img_embedding_dim)  
    self.img_lin12 = nn.Linear(64, self.img_embedding_dim) 

    self.img_change_dim = nn.Linear(self.img_embedding_dim + 10, 64 * 1 * 1)

    self.img_decoder = nn.Sequential(
      nn.ConvTranspose2d(64, 32, kernel_size = 4, stride = 2),
      nn.ReLU(True),
      nn.ConvTranspose2d(32, 16, kernel_size = 6, stride = 2),
      nn.ReLU(True),
      nn.ConvTranspose2d(16, 1, kernel_size = 6, stride = 2),
      nn.Sigmoid())

    self.label_decoder = nn.Sequential(
      nn.Linear(64, 128),
      nn.ReLU(),
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Linear(128, 10),
      nn.Sigmoid())

  def reparameterize(self, mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = Variable(std.data.new(std.size()).normal_())
    return eps.mul(std).add_(mu)

  def forward(self, x, l_onehots):
    
    # encode 
    x_e = self.img_encoder(x)
    x_e = x_e.view(x.size(0), -1)
    mu = self.img_lin11(x_e)
    logvar = self.img_lin12(x_e)

    z = self.reparameterize(mu, logvar)

    # decode
    x_d = self.img_change_dim(torch.cat((l_onehots, z), dim = 1))
    x_d_common = x_d.view(x.size(0), 64, 1, 1)
    x_d = self.img_decoder(x_d_common)

    y_d = self.label_decoder(x_d_common.view(x.size(0), 64))

    return x_d, y_d, z, mu, logvar

# Train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('jmvae.pt').to(device)

def infer(model):
  model.eval()
  for data in test_loader:
    x, y = data
    x, y = x.to(device), y.to(device, dtype = torch.float32)

    l_onehots_ = torch.zeros(x.size(0), 10).to(device)
    for i in range(x.size(0)):
      l_onehots_[i, int(y[i])] = 1

    #l_onehots = torch.rand(x.size(0), 10).to(device)
    #pads = torch.rand(x.size(0), 28 * 28 - 10).to(device)
    #l_e = torch.cat((l_onehots, pads), dim = 1)
    #l_e = l_e.view(l_e.size(0), 1, 28, 28)
    #x_e = torch.cat((x, l_e), dim = 1) 
      
    l_onehots = l_onehots_
    pads = torch.zeros(x.size(0), 28 * 28 - 10).to(device)
    l_e = torch.cat((l_onehots, pads), dim = 1)
    l_e = l_e.view(l_e.size(0), 1, 28, 28)
    x_e = torch.cat((torch.rand_like(x).to(device), l_e), dim = 1)
    # ===================forward=====================
    x_d, y_d, z, mu, logvar = model(x_e, l_onehots)
    print('Label:', y, 'Predicted:', torch.argmax(y_d))

    x_d = x_d.detach().cpu().numpy()[0, 0, :, :]
    pixels = x_d
    plt.imshow(pixels, cmap = 'gray')
    plt.savefig('sample_jmvae.png')

    break
   
infer(model) 
