
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(1)

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

# print(train_data.train_data[0])  #28*28 range(0,255)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)



G = nn.Sequential(                      # Generator
    nn.Linear(N_noise, 500),
    nn.ReLU(),
    nn.Linear(500, 1000),
    nn.ReLU(),
    nn.Linear(1000, 600),
    nn.ReLU(),
    nn.Linear(600, 28*28),
    nn.Sigmoid(),
)

D = nn.Sequential(                      # Discriminator
    nn.Linear(28*28, 500),
    nn.ReLU(),
    nn.Linear(500, 1000),
    nn.ReLU(),
    nn.Linear(1000, 200),
    nn.ReLU(),
    nn.Linear(200, 1),
    nn.Sigmoid(),
)

opt_D = torch.optim.Adam(D.parameters(), lr=0.0002)
opt_G = torch.optim.Adam(G.parameters(), lr=0.0002)

plt.ion()
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        images = x.squeeze().view(-1,28*28)
        real_image = Variable(images)

        G_ideas = Variable(torch.randn(BATCH_SIZE, N_noise))
        G_fakeimage = G(G_ideas)

        prob0 = D(real_image)
        prob1 = D(G_fakeimage)

        D_loss = - torch.mean(torch.log(prob0) + torch.log(1. - prob1))
        G_loss = torch.mean(torch.log(1. - prob1))

        opt_D.zero_grad()
        D_loss.backward(retain_variables=True)
        opt_D.step()

        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()

        if step % 100 == 0:
            plt.cla()
            plt.imshow(G_fakeimage.data[1].view(28,28).numpy(), cmap='gray')
            plt.draw();
            plt.pause(0.01)

plt.ioff()
plt.show()