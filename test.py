import torch
from dataloader.datalo import testdata
from models.generator import Stage_1 as model
import os, sys, gc, argparse, numpy as np
from torchvision.utils import save_image
from torch.autograd import Variable
from torchvision.utils import save_image
from binascii import a2b_base64

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cimage', type=str, default='./cimg.jpg', help='corrupted image')
    parser.add_argument('--eimage', type=str, default='./eimg.jpg', help='edge map')
    parser.add_argument("--simage", type=str, default='./', help='save image')
    opt = parser.parse_args()
    return opt
def get_img():
    cimg = open("./cimg.txt", "br")
    eimg = open("./eimg.txt", "br")
    binary_data = a2b_base64(cimg.read())
    fd = open('cimg.jpg', 'wb')
    fd.write(binary_data)
    fd.close()
    binary_data = a2b_base64(eimg.read())
    fd = open('eimg.jpg', 'wb')
    fd.write(binary_data)
    fd.close()


def get_res(opt):
    datalo = testdata()
    cimage = opt.cimage
    eimage = opt.eimage
    simage = opt.simage
    get_img()
    net = model(3,3)
    print("loading model")
    net.cpu()
    net.load_state_dict(torch.load("./models/Gan_21.pth",map_location='cpu'))#, map_location='cpu')
    net.eval()
    
    cimg,eimg = datalo.getImages(cimage,eimage)
    cimg,eimg = Variable(cimg.cpu()),Variable(eimg.cpu())
    
    gen = net(cimg,eimg.float())
    pic = (torch.cat([gen], dim=0).data + 1) / 2.0
    save_dir = simage
    save_image(pic, '{}/simage.jpg'.format(save_dir), nrow=1)
    
def main():
    opt = get_opt()
    get_res(opt)
    
    
    
if __name__ == "__main__":
    main()
    