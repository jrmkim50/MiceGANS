from __future__ import print_function
import argparse
import os
from os import listdir
from os.path import join
import numpy as np

import torch
import torchvision.transforms as transforms

import nibabel as nib
from utils import is_image_file, load_img, save_img

# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
parser.add_argument('--nepochs', type=int, default=200, help='saved model of which epochs')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if opt.cuda else "cpu")

model_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, opt.nepochs)

net_g = torch.load(model_path).to(device)

if opt.direction == "a2b":
    image_dir = "dataset/{}/test/a/".format(opt.dataset)
else:
    image_dir = "dataset/{}/test/b/".format(opt.dataset)

# image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

NiftyFilenames = [file for file in listdir(image_dir) if file[0] != '.']
image_filenames = []
# for file in NiftyFilenames:
#     vol = nib.load(join(image_dir, file))
#     for i in range(vol.shape[2]):
#         image_filenames.append((file, i))
for file in NiftyFilenames:
    vol = nib.load(join(image_dir, file))
    for i in range(int((opt.input_nc-1)/2), vol.shape[2]-int((opt.input_nc-1)/2)):
        image_filenames.append((file, i))

# transform_list = [transforms.ToTensor(),
#                   transforms.Normalize(0.5, 0.5)]
transform_list = [transforms.ToTensor()]

transform = transforms.Compose(transform_list)

if not os.path.exists(os.path.join("result", opt.dataset)):
    os.makedirs(os.path.join("result", opt.dataset))

vol = []
vol_name = ''
for image_name in image_filenames:
    # img = load_img(image_dir + image_name)
    # img = nib.load(join(image_dir, image_name[0])).get_fdata()[:, :, image_name[1]]
    img = nib.load(join(image_dir, image_name[0])).get_fdata()[:, :,
        image_name[1] - int((opt.input_nc - 1) / 2):image_name[1] + 1 + int((opt.input_nc - 1) / 2)]
    img = transform(img)
    input = img.unsqueeze(0).to(device)
    out = net_g(input.float())
    out_img = out.detach().squeeze(0).cpu().float().numpy().squeeze()

    if image_name[0] != vol_name:
        if len(vol) == 0:
            if opt.input_nc > 1:
                vol = np.zeros_like(out_img)
                for i in range(int((opt.input_nc - 1) / 2 -1)):
                    vol = np.dstack((vol, np.zeros_like(out_img)))#for i in range(int((opt.input_nc - 1) / 2):
                vol = np.dstack((vol, out_img))
            else:
                vol = out_img
            vol_name = image_name[0]
        else:
            for i in range(int((opt.input_nc - 1) / 2 )):
                vol = np.dstack((vol, np.zeros_like(out_img)))  # for i in range(int((opt.input_nc - 1) / 2)
            NiftiImg = nib.Nifti1Image(vol, np.eye(4))
            # nib.save(NiftiImg, "result/{}/{}.nii.gz".format(opt.dataset, image_name[0].split('.')[0]))
            nib.save(NiftiImg, "result/{}/{}.nii.gz".format(opt.dataset, vol_name.split('.')[0]))
            print("result/{}/{}.nii.gz".format(opt.dataset, vol_name.split('.')[0]))
            if opt.input_nc > 1:
                vol = np.zeros_like(out_img)
                for i in range(int((opt.input_nc - 1) / 2 - 1)):
                    vol = np.dstack((vol, np.zeros_like(out_img)))  # for i in range(int((opt.input_nc - 1) / 2):
                vol = np.dstack((vol, out_img))
            else:
                vol = out_img
            vol_name = image_name[0]
    else:
        vol = np.dstack((vol, out_img))

    # NiftiImg = nib.Nifti1Image(out_img, np.eye(4))
    # nib.save(NiftiImg, "result/{}/{}_{}.nii.gz".format(opt.dataset, image_name[0].split('.')[0], image_name[1]))
    # save_img(out_img, "result/{}/{}".format(opt.dataset, image_name))
    print("result/{}/{}_{}.nii.gz".format(opt.dataset, image_name[0].split('.')[0], image_name[1]))

NiftiImg = nib.Nifti1Image(vol, np.eye(4))
nib.save(NiftiImg, "result/{}/{}.nii.gz".format(opt.dataset, image_name[0].split('.')[0]))
print("result/{}/{}.nii.gz".format(opt.dataset, image_name[0].split('.')[0]))