from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from visdom import Visdom

import generator
import discriminator

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main', xlabel='Epochs'):
        self.viz = Visdom()
        self.env = env_name
        self.xlabel = xlabel
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]),
                Y=np.array([y,y]),
                env=self.env,
                opts=dict(
                    legend=[split_name],
                    title=title_name,
                    xlabel=self.xlabel,
                    ylabel=var_name
                )
            )
        else:
            self.viz.line(X=np.array([x]),
                Y=np.array([y]), env=self.env,
                win=self.plots[var_name],
                name=split_name,
                update = 'append')


batchSize = 64
imageSize = 64

transform = transforms.Compose(
    [transforms.Resize(imageSize),
     transforms.ToTensor(),
     transforms.Normalize(
         (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

dataset = dset.CIFAR10(root = '../data',
                       download = True,
                       transform = transform)
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size = batchSize,
                                         shuffle = True,
                                         num_workers = 4)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

generator_net = generator.Generator() 
generator_net.apply(weights_init)

discriminator_net = discriminator.Discriminator()
discriminator_net.apply(weights_init)

criterion = nn.BCELoss()
gen_optimizer = optim.Adam(generator_net.parameters(),
                        lr = 0.0002,
                        betas = (0.5, 0.999))
dis_optimizer = optim.Adam(discriminator_net.parameters(),
                        lr = 0.0002,
                        betas = (0.5, 0.999))

epoch_plotter = VisdomLinePlotter(env_name='Train Plots')

fake_image_vis = Visdom(env='Train Plots')
vis = Visdom(env='Train Plots')
iteration_plotter = VisdomLinePlotter(env_name='Train Plots', xlabel='Iteration')

img_id = None
for epoch in range(25):
    vis.text('Epoc #' + str(epoch))
    
    for i, data in enumerate(dataloader, 0):
        discriminator_net.zero_grad()
        real, _ = data
        input = Variable(real)
        target = Variable(torch.ones(input.size()[0]))
        output = discriminator_net(input)
        dis_err_real = criterion(output, target)
        
        noise = Variable(torch.randn(input.size()[0], 100, 1, 1))
        fake = generator_net(noise)
        target = Variable(torch.zeros(input.size()[0]))
        output = discriminator_net(fake.detach())
        dis_err_fake = criterion(output, target)
        
        dis_err = dis_err_real + dis_err_fake
        dis_err.backward()
        dis_optimizer.step()
        
        generator_net.zero_grad()
        target = Variable(torch.ones(input.size()[0]))
        output = discriminator_net(fake)
        gen_err = criterion(output, target)
        gen_err.backward()
        gen_optimizer.step()

        gen_id = iteration_plotter.plot('gen_loss', 'train', 'Generator Loss', i, gen_err.data[0])
        dis_id = iteration_plotter.plot('dis_loss', 'train', 'Discriminator Loss', i, dis_err.data[0])
        
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 25, i, len(dataloader), dis_err.data[0], gen_err.data[0]))
        if i % 100 == 0:
            vutils.save_image(real, '%s/real_samples.png' % "./results", normalize = True)
            fake = generator_net(noise)
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize = True)
            if img_id:
                vis.close(win=img_id)
            img_id = fake_image_vis.images(fake.data)
    vis.close(win=gen_id)
    vis.close(win=dis_id)
    epoch_plotter.plot('gen_loss', 'train', 'Generator Loss', epoch, gen_err.data.mean())
    epoch_plotter.plot('dis_loss', 'train', 'Discriminator Loss', epoch, dis_err.data.mean())