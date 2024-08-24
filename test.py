import torch
import torch.nn as nn
import torch.optim as optimizer
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


device='cuda' if torch.cuda.is_available() else 'cpu'

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


def get_noise(num_samples,num_features=64,device=device):
  noise=torch.randn(num_samples,num_features).to(device)
  return noise.view(len(noise),num_features,1,1)

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,)),
    transforms.Resize(size=(32,32))]
)
data=datasets.MNIST('data',transform=transform,download=True)

Dataloader=DataLoader(data,batch_size=128,shuffle=True)


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

generator=gen(64,1,64).to(device)
generator.load_state_dict(torch.load('model_save_gen.pt'))
real,_ =next(iter(Dataloader))
# torch.manual_seed(42)
noise3=get_noise(len(real))
generated=generator(noise3)
print(generated.shape)
visualize(generated)