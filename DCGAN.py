
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np

EPOCH = 2      
BATCH_SIZE = 50 
LR = 0.001      
N_noise = 100  


train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=False,
)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.layn.Sequential(nn.ConvTranspose2d(100,64*4,kernel_size=4),
                                 nn.BatchNorm2d(64*4),
                                 nn.ReLU())
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(64*4,64*2,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(64*2),
                                 nn.ReLU())
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(64*2,64,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU())
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(64,1,kernel_size=4,stride=2,padding=3),
                                 nn.Tanh())
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        self.layer1 = nn.Sequential(nn.Conv2d(1,64,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(64),
                                 nn.LeakyReLU(0.2,inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(64,64*2,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(64*2),
                                 nn.LeakyReLU(0.2,inplace=True))
        self.layer3 = nn.Sequential(nn.Conv2d(64*2,64*4,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(64*4),
                                 nn.LeakyReLU(0.2,inplace=True))
        self.layer4 = nn.Sequential(nn.Conv2d(64*4,1,kernel_size=3,stride=1,padding=0),
                                 nn.Sigmoid())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out
D = Discriminator()
G = Generator()
loss_func = nn.BCELoss()
optimizerD = torch.optim.Adam(D.parameters(),lr=0.0002, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(G.parameters(),lr=0.0002, betas=(0.5, 0.999))

plt.ion()
for epoch in range(EPOCH):
    for step,(images,_) in enumerate(train_loader):
        D.zero_grad()
        real_images = Variable(images)
        d_real = D(real_images)
        d_realloss = loss_func(d_real, Variable(torch.ones(BATCH_SIZE,1,1,1)))
        d_realloss.backward()
        G_ideas = Variable(torch.randn(BATCH_SIZE, N_noise,1,1))
        G_fakeimage = G(G_ideas)
        d_fake = D(G_fakeimage.detach())
        d_fakeloss = loss_func(d_fake, Variable(torch.zeros(BATCH_SIZE,1,1,1)))
        d_fakeloss.backward()
        d_loss = d_realloss + d_fakeloss
        optimizerD.step()
        G.zero_grad()
        out = D(G_fakeimage)
        g_loss = loss_func(out, Variable(torch.ones(BATCH_SIZE,1,1,1)))
        g_loss.backward()
        optimizerG.step()
        if step % 50 == 0:
            plt.cla()
            plt.imshow(G_fakeimage.data[1].view(28,28).numpy(), cmap='gray')
            plt.draw()
            plt.pause(0.01)


plt.ioff()
plt.show()
torch.save(G.state_dict(),"dcgan_g.pkl")