# -*- coding: utf-8 -*-
"""GAN_CONV.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1n3GjdBsaAzBkv_NghCTVS_Rd3iyIVUIM
"""
import torch
import torch.nn as nn
import torch.optim as optimizer
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device='cuda' if torch.cuda.is_available() else 'cpu'

"""GENERATOR BLOCK AND GENERATOR"""

def gen_block(input_features,output_features,kernel_size,stride,padding):
  return nn.Sequential(
      nn.ConvTranspose2d(input_features,output_features,kernel_size,stride,padding),
      nn.BatchNorm2d(output_features),
      nn.ReLU(inplace=True)
  )

class gen(nn.Module):
  def __init__(self,input_features,output_features,hidden_units):
    super(). __init__()
    self.layer_stack=nn.Sequential(
        gen_block(input_features,hidden_units*2,kernel_size=4,stride=1,padding=0),
        gen_block(hidden_units*2,hidden_units,kernel_size=4,stride=2,padding=1),
        gen_block(hidden_units,hidden_units//2,kernel_size=4,stride=2,padding=1),
        nn.ConvTranspose2d(hidden_units//2,output_features,kernel_size=4,stride=2,padding=1),
        nn.Tanh()
    )
  def forward(self,noise):
    return self.layer_stack(noise)

"""NOISE GENERATOR"""

def get_noise(num_samples,num_features=64,device=device):
  noise=torch.randn(num_samples,num_features).to(device)
  return noise.view(len(noise),num_features,1,1)

"""DISCRIMINATOR"""

def disc_block(input_features,output_features,kernel_size=4,stride=2):
  return nn.Sequential(
      nn.Conv2d(input_features,output_features,kernel_size,stride),
      nn.BatchNorm2d(output_features),
      nn.LeakyReLU(0.2,inplace=True)
  )

class disc(nn.Module):
  def __init__(self,input_features,hidden_units,output_features):
    super(). __init__()
    self.layer_stack=nn.Sequential(
        disc_block(input_features,hidden_units),
        disc_block(hidden_units,hidden_units*2),
        nn.Conv2d(hidden_units*2,1,kernel_size=4,stride=2)
    )
  def forward(self,image):
    pred=self.layer_stack(image)
    return pred.view(len(pred),-1)

"""SETTING UP THE LOSS FUNCTIONS AND DATALOADER"""

beta1=0.5
beta2=0.999

loss_fn=nn.BCEWithLogitsLoss()
generator=gen(64,1,64)
gen_optim=optimizer.Adam(generator.parameters(),lr=0.001,betas=(beta1,beta2))
discriminator=disc(1,16,1)
disc_optim=optimizer.Adam(discriminator.parameters(),lr=0.001,betas=(beta1,beta2))

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,)),
    transforms.Resize(size=(32,32))]
)
data=datasets.MNIST('data',transform=transform,download=True)

Dataloader=DataLoader(data,batch_size=128,shuffle=True)

"""VISUALIZING THE DATA"""

def visualize(img_tensor,num_images=25,size=(1,28,28)):
  img_tensor=(img_tensor+1)/2
  fig=plt.figure(figsize=(10,10))
  images=img_tensor[:num_images]
  for i in range(num_images):
    img=fig.add_subplot(5,5,i+1)
    # print(images[i].shape)
    img.imshow(images[i].permute(1,2,0).detach().cpu(),cmap='gray')
    plt.axis('off')
  plt.show()

import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

"""TRAINING LOOP"""

epochs=200
generator.to(device)
discriminator.to(device)
for epoch in range(epochs):
  disc_loss=0
  gen_loss=0

  for real,_ in Dataloader:
    real=real.to(device)
    # disc loss
    noise=get_noise(len(real))
    # print(noise.shape)
    fake_img=generator(noise)
    # print(fake_img.shape)
    fake_pred=discriminator(fake_img.detach())
    fake_loss=loss_fn(fake_pred,torch.zeros_like(fake_pred))
    real_pred=discriminator(real)
    real_loss=loss_fn(real_pred,torch.ones_like(real_pred))
    loss=(fake_loss+real_loss)/2
    disc_loss+=loss.item()
    disc_optim.zero_grad()

    loss.backward()

    disc_optim.step()

    # gen loss
    noise2=get_noise(len(real))
    fake_img2=generator(noise)
    fake_pred2=discriminator(fake_img2)
    fake_loss=loss_fn(fake_pred2,torch.ones_like(fake_pred2))
    gen_loss+=fake_loss.item()

    gen_optim.zero_grad()

    fake_loss.backward()

    gen_optim.step()

  print(f'epoch {epoch} | gen_loss {fake_loss/len(Dataloader):.4f} | disc_loss {loss/len(Dataloader):.4f}')
  writer.add_scalar('gen loss',fake_loss,epoch)
  writer.add_scalar('disc loss',loss,epoch)

  if epoch%10==0:
    # torch.manual_seed(42)
    noise3=get_noise(len(real))
    generated=generator(noise3)
    # print(generated.shape)
    visualize(real)
    visualize(generated)
    torch.save(generator.state_dict(),'model_save_gen.pt')
    torch.save(discriminator.state_dict(),'model_save_disc.pt')

writer.close()
# real,_ =next(iter(Dataloader))
# # torch.manual_seed(42)
# noise3=get_noise(len(real))
# generated=generator(noise3)
# print(generated.shape)
# visualize(generated)