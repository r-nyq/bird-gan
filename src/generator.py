import torch.nn as nn

class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(200, 512, 4, 1, 0, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 384, 4, 2, 1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(384, 192, 4, 2, 1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
nn.Tanh()
        )
        
    def forward(self, input):
        output = self.main(input)
        return output