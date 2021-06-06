from os import listdir
from os.path import join
import numpy as np
import random

from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import nibabel as nib

from utils import is_image_file, load_img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, direction, input_nc):
        super(DatasetFromFolder, self).__init__() #this is the way do parameterless super() call
        self.direction = direction
        self.input_nc = input_nc
        self.a_path = join(image_dir, "a")
        self.b_path = join(image_dir, "b")
        NiftyFilenames = [x for x in listdir(self.a_path) if x[0]!='.']
        slices = []
        for file in NiftyFilenames:
            vol = nib.load(join(self.a_path, file))
            for i in range(int((self.input_nc-1)/2), vol.shape[2]-int((self.input_nc-1)/2)):
                slices.append((file, i))
        self.image_filenames = slices

        # transform_list = [transforms.ToTensor(),
        #                   transforms.Normalize(0.5, 0.5)]
        transform_list = [transforms.ToTensor()]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        # a = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
        # b = Image.open(join(self.b_path, self.image_filenames[index])).convert('RGB')
        ind = self.image_filenames[index][1]
        int((self.input_nc-1)/2)
        a = nib.load(join(self.a_path, self.image_filenames[index][0])).get_fdata()[:, :,
            ind]
        # a = np.swapaxes(a, 0, 2)
        b = nib.load(join(self.b_path, self.image_filenames[index][0])).get_fdata()[:, :,
            ind-int((self.input_nc-1)/2):ind+1+int((self.input_nc-1)/2)]
        # b = np.swapaxes(b, 0, 2)
        # a = a.resize((286, 286), Image.BICUBIC)
        # b = b.resize((286, 286), Image.BICUBIC)
        a = transforms.ToTensor()(a)
        b = transforms.ToTensor()(b)
        # w_offset = random.randint(0, max( 0, 286 - 256 - 1))
        # h_offset = random.randint(0, max(0, 286 - 256 - 1))
    
        # a = a[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
        # b = b[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
    
        # a = transforms.Normalize(0.5, 0.5)(a)
        # b = transforms.Normalize(0.5, 0.5)(b)

        # "Im not sure what the following part is doing"
        # if random.random() < 0.5:
        #     idx = [i for i in range(a.size(2) - 1, -1, -1)]
        #     idx = torch.LongTensor(idx)
        #     a = a.index_select(2, idx)
        #     b = b.index_select(2, idx)

        if self.direction == "a2b":
            return a, b
        else:
            return b, a

    def __len__(self):
        return len(self.image_filenames)
