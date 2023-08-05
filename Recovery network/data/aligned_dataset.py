### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path

import util.util
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset, make_sortedDataset
from PIL import Image
from PIL import ImageFilter
import numpy as np
import torchvision.transforms.functional as F
from . import sobel
import random

from scipy import ndimage


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.edge = sobel.GradLayer()

        self.scen = opt.scen

        ### input A (label maps)
        dir_A = 'Input_S'  # Input / Input_S / Input_WO_N
        self.dir_A = os.path.join(opt.dataroot, dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)
        dir_B = 'GT' # GT
        self.dir_B = os.path.join(opt.dataroot, dir_B)
        self.B_paths = sorted(make_dataset(self.dir_B))

        # dir_C = 'GTT_1'
        # self.dir_C = os.path.join(opt.dataroot, dir_C)
        # self.C_paths = sorted(make_dataset(self.dir_C))
        # self.dataset_size = len(self.A_paths)

    def __modsize(self, img, modsize):
        W, H = img.size
        H_r, W_r = H % modsize, W % modsize
        return img.crop((0, 0, W - W_r, H - H_r))

    def __gen_kernel(self, k_size=np.array([25, 25]), min_var=0.6, max_var=12.):
        """"
        # modified version of https://github.com/assafshocher/BlindSR_dataset_generator
        # Kai Zhang
        # min_var = 0.175 * sf  # variance of the gaussian kernel will be sampled between min_var and max_var
        # max_var = 2.5 * sf
        """

        sf = random.choice([1, 2, 3, 4])
        scale_factor = np.array([sf, sf])
        # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
        lambda_1 = min_var + np.random.rand() * (max_var - min_var)
        lambda_2 = min_var + np.random.rand() * (max_var - min_var)
        theta = np.random.rand() * np.pi  # random theta
        noise = 0  # -noise_level + np.random.rand(*k_size) * noise_level * 2

        # Set COV matrix using Lambdas and Theta
        LAMBDA = np.diag([lambda_1, lambda_2])
        Q = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        SIGMA = Q @ LAMBDA @ Q.T
        INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

        # Set expectation position (shifting kernel for aligned image)
        MU = k_size // 2 - 0.5 * (scale_factor - 1)  # - 0.5 * (scale_factor - k_size % 2)
        MU = MU[None, None, :, None]

        # Create meshgrid for Gaussian
        [X, Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
        Z = np.stack([X, Y], 2)[:, :, :, None]

        # Calcualte Gaussian for every pixel of the kernel
        ZZ = Z - MU
        ZZ_t = ZZ.transpose(0, 1, 3, 2)
        raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)

        # shift the kernel so it will be centered
        # raw_kernel_centered = kernel_shift(raw_kernel, scale_factor)

        # Normalize the kernel and return
        # kernel = raw_kernel_centered / np.sum(raw_kernel_centered)
        kernel = raw_kernel / np.sum(raw_kernel)
        return kernel


    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A = Image.open(A_path).convert('RGB')
        A = self.__modsize(A, 32)
        (ow, oh) = A.size

        params = get_params(self.opt, A.size)
        transform_A = get_transform(self.opt, params, resize=False)
        A_tensor = transform_A(A)

        ### input B (real images)
        B_path = self.B_paths[index]
        B = Image.open(B_path).convert('RGB')
        B = self.__modsize(B, 32)

        transform_B = get_transform(self.opt, params, resize=False)
        B_tensor = transform_B(B)
        image_edge = self.edge(B_tensor.unsqueeze(0)).squeeze(0)

        ### input C (thumbnail images)
        if ow > oh:
            ratio = oh / ow
            width = 160
            height = int(width * ratio)

        else:
            ratio = ow / oh
            height = 160
            width = int(height * ratio)

        C = B.resize((width, height),  Image.BICUBIC)
        C_resize_1 = C.resize((ow, oh), Image.BICUBIC)
        C_resize_2 = C.resize((ow // 2 ** self.scen, oh // 2 ** self.scen), Image.NEAREST)

        transform_C = get_transform(self.opt, params, resize=False)
        B_thumbnail_first_tensor = transform_C(C_resize_1)
        B_thumbnail_second_tensor = transform_C(C_resize_2)


        input_dict = {'label': A_tensor, 'image': B_tensor, 'path': A_path, 'gt_image_edge': image_edge,
                      'gt_thumbnail': B_thumbnail_first_tensor,
                      'gt_thumbnail_original': B_thumbnail_second_tensor}

        return input_dict


    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset'

    def __scale_width(self, img, target_width, method=Image.BICUBIC):
        ow, oh = img.size
        if (ow == target_width):
            return img
        w = target_width
        h = int(target_width * oh / ow)
        return img.resize((w, h), method)

    def image_concat(self, image):
        width, height = image.size
        new_image = Image.new(image.mode, (width * 2, height))
        new_image.paste(image, box=(0, 0))
        new_image.paste(image, box=(width, 0))
        return new_image


if __name__ == '__main__':
    import scipy.io as io
    import matplotlib.pyplot as plt
    kernels = io.loadmat('/home/liu/code/EPDN/kernels_12.mat')['kernels']  # for validation
    k = kernels[0, 0].astype(np.float64)  # validation kernel
    k /= np.sum(k)
    plt.imshow(k, cmap='gray')
    plt.show()