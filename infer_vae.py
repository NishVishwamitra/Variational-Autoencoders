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
    torchvision.transforms.ToTensor()])),
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
    #x_e = self.img_encoder(x)
    #x_e = x_e.view(x.size(0), -1)
    #mu = self.img_lin11(x_e)
    #logvar = self.img_lin12(x_e)

    #z = self.reparameterize(mu, logvar)
    z = torch.rand(1, 100).to(device)

    # decode
    x_d = self.img_change_dim(z)
    x_d = x_d.view(x.size(0), 64, 1, 1)
    x_d = self.img_decoder(x_d)

    return x_d, z

# Train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('vae.pt').to(device)

def infer(model):
  model.eval()
  with torch.no_grad():
    for data in test_loader:
      x, y = data
      x, y = x.to(device), y.to(device, dtype = torch.float32)
      x_d, z = model(x)
      x_d = x_d.detach().cpu().numpy()
      pixels = x_d.reshape((28, 28))
      plt.imshow(pixels, cmap = 'gray')
      plt.savefig('sample.png')
      break


infer(model)
