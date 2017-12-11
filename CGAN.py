
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
    labels = (a-1) > 0.5
    realpoint = torch.from_numpy(realpoint).float()
    labels = torch.from_numpy(labels.astype(np.float32))
    return Variable(realpoint) , Variable(labels)


G = nn.Sequential(  # Generator
    nn.Linear(5+1, 128), 
    nn.ReLU(),
    nn.Linear(128, 15),
)

D = nn.Sequential(  # Discriminator
    nn.Linear(15+1, 128), 
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(), 
)

opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

plt.ion()
for step in range(10000):

    realdatas,labels= realdata()
    G_noise = Variable(torch.randn(64, 5))
    G_inputs = torch.cat((G_noise, labels), 1)
    G_fakedata = G(G_inputs) 
	
    D_inputs0 = torch.cat((realdatas, labels), 1)   
    D_inputs1 = torch.cat((G_fakedata, labels), 1)
    prob0 = D(D_inputs0)                                                  
    prob1 = D(D_inputs1)         
	
 
    D_loss = - torch.mean(torch.log(prob0) + torch.log(1. - prob1))
    G_loss = torch.mean(torch.log(1. - prob1))

    opt_D.zero_grad()
    D_loss.backward(retain_variables=True)
    opt_D.step()

    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()

    if step % 200 == 0:  # plotting
        plt.cla()
        plt.plot(point[0], G_fakedata.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting',)
        bound = [0, 0.5] if labels.data[0, 0] == 0 else [0.5, 1]
        plt.plot(point[0], 2 * np.power(point[0], 2) + bound[1], c='#74BCFF', lw=3, label='max bound')
        plt.plot(point[0], 1 * np.power(point[0], 2) + bound[0], c='#FF9359', lw=3, label='min bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob0.data.numpy().mean(), fontdict={'size': 15})
        plt.text(-.5, 1.7, 'Class = %i' % int(labels.data[0, 0]), fontdict={'size': 15})
        plt.ylim((0, 3));plt.legend(loc='upper left', fontsize=8);plt.draw();plt.pause(0.1)

plt.ioff()
plt.show()
torch.save(G.state_dict(),"generater.pkl")