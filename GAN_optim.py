
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)  
np.random.seed(1)

LR_G = 0.0001
LR_D = 0.0001

point = np.vstack([np.linspace(-1, 1, 15) for _ in range(64)])

def realdata():
    a = np.random.uniform(1, 2, size=64)[:, np.newaxis]
    realpoint = a * np.power(point,2) + (a - 1)
    realpoint = torch.from_numpy(realpoint).float()
    return Variable(realpoint)


G = nn.Sequential(  # Generator
    nn.Linear(5, 128),
    nn.ReLU(),
    nn.Linear(128, 15),
)

D = nn.Sequential(  # Discriminator
    nn.Linear(15, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(),
)



opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)
loss_func = nn.BCELoss()

plt.ion()

for step in range(10000):

    opt_D.zero_grad()
    # 1A
    realdatas = realdata()
    d_real = D(realdatas)
    d_realloss = loss_func(d_real,Variable(torch.ones(64,1)))
    d_realloss.backward()
    #1B
    G_noise = Variable(torch.randn(64, 5))
    G_fakedata = G(G_noise)
    d_fake = D(G_fakedata.detach())
    d_fakeloss = loss_func(d_fake,Variable(torch.zeros(64,1)))
    d_fakeloss.backward()
    opt_D.step()

    #2
    opt_G.zero_grad()

    g_fake = D(G_fakedata)
    g_loss = loss_func(g_fake,Variable(torch.ones(64,1)))
    g_loss.backward()
    opt_G.step()

    if step % 50 == 0:
        plt.cla()
        plt.plot(point[0], G_fakedata.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting', )
        plt.plot(point[0], 2 * np.power(point[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(point[0], 1 * np.power(point[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f' % d_real.data.numpy().mean(),
                 fontdict={'size': 15})
        plt.ylim((0, 3));
        plt.legend(loc='upper left', fontsize=8);
        plt.draw();
        plt.pause(0.01)

plt.ioff()
plt.show()
