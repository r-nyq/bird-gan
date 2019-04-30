import generator
import argparse

import torch
from torch.autograd import Variable
import torchvision.utils as vutils

parser = argparse.ArgumentParser()
parser.add_argument("model", help="Generator model path",
	type=str)
parser.add_argument("--image_path", help="Path for image output",
	type=str, default='./generated_images/generated_img.png')

args = parser.parse_args()

model = torch.load(args.model)
model.eval()

noise = Variable(torch.randn(1, 400, 1, 1)).cuda()
fake = model(noise).cuda()
vutils.save_image(fake, args.image_path ,normalize = True)