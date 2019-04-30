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

import plotter
import generator
import discriminator

cuda = torch.device('cuda')

batchSize = 64
imageSize = 128

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
std=[0.5, 0.5, 0.5])

transform = transforms.Compose([
        transforms.Resize((imageSize, imageSize)),
        transforms.ToTensor(),
        normalize
])

dataset = dset.ImageFolder(
    root = 'data/images',
    transform=transform)
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
generator_net.cuda()
generator_net.apply(weights_init)

discriminator_net = discriminator.Discriminator()
discriminator_net.cuda()
discriminator_net.apply(weights_init)

criterion = nn.BCELoss()
gen_optimizer = optim.Adam(generator_net.parameters(),
                        lr = 0.0002,
                        betas = (0.5, 0.999))
dis_optimizer = optim.Adam(discriminator_net.parameters(),
                        lr = 0.0002,
                        betas = (0.5, 0.999))

epoch_plotter = plotter.EpochPlotter(env_name='Train Plots')

fake_image_vis = Visdom(env='Train Plots')
vis = Visdom(env='Train Plots')

img_id = None
img_id2 = None
for epoch in range(5000):
    text_id = vis.text('Epoc #' + str(epoch))
    epoch_gen_err = []
    epoch_dis_err = []
    for i, data in enumerate(dataloader, 0):
        discriminator_net.zero_grad()
        real, _ = data
        input = Variable(real).cuda()
        target = Variable(torch.ones(input.size()[0])).cuda()
        output = discriminator_net(input).cuda()
        dis_err_real = criterion(output, target).cuda()
        
        noise = Variable(torch.randn(input.size()[0], 400, 1, 1)).cuda()
        fake = generator_net(noise).cuda()
        target = Variable(torch.zeros(input.size()[0])).cuda()
        output = discriminator_net(fake.detach()).cuda()
        dis_err_fake = criterion(output, target).cuda()
        
        dis_err = dis_err_real + dis_err_fake
        dis_err.backward()
        dis_optimizer.step()
        
        generator_net.zero_grad()
        target = Variable(torch.ones(input.size()[0])).cuda()
        output = discriminator_net(fake).cuda()
        gen_err = criterion(output, target).cuda()
        gen_err.backward()
        gen_optimizer.step()

        epoch_gen_err.append(gen_err.item())
        epoch_dis_err.append(dis_err.item())
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 5000, i, len(dataloader), dis_err.item(), gen_err.item()))
        if i % 100 == 0:
            vutils.save_image(real, '%s/real_samples.png' % "./results", normalize = True)
            fake = generator_net(noise)
            vutils.save_image(fake, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize = True)
            if img_id:
                vis.close(win=img_id)
                vis.close(win=img_id2)
            img_id = fake_image_vis.images(fake.detach().cpu().numpy())
            img_id2 = fake_image_vis.images(real.detach().cpu().numpy())

    vis.close(win=text_id)

    epoch_plotter.plot('epoch_gen_loss', 'train', 'Generator Epoch Loss', epoch, np.mean(epoch_gen_err))
    epoch_plotter.plot('epoch_dis_loss', 'train', 'Discriminator Epoch Loss', epoch, np.mean(epoch_dis_err))
