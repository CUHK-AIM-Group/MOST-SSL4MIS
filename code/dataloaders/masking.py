import random
import warnings

import kornia
import numpy as np
import torch
from einops import repeat
from torch import nn, Tensor
from torch.nn import functional as F

warnings.filterwarnings("ignore", category=DeprecationWarning)

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def strong_transform(param, data):
    data = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=data)
    data = gaussian_blur(blur=param['blur'], data=data)
    return data


def denorm(img, mean, std):
    return img.mul(std).add(mean)


def renorm(img, mean, std):
    return img.sub(mean).div(std)


def color_jitter(color_jitter, mean, std, data, s=.25, p=.2):
    # s is the strength of colorjitter
    if color_jitter > p:
        mean = torch.as_tensor(mean, device=data.device)
        mean = repeat(mean, 'C -> B C 1 1', B=data.shape[0], C=3)
        std = torch.as_tensor(std, device=data.device)
        std = repeat(std, 'C -> B C 1 1', B=data.shape[0], C=3)
        if isinstance(s, dict):
            seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
        else:
            seq = nn.Sequential(
                kornia.augmentation.ColorJitter(
                    brightness=s, contrast=s, saturation=s, hue=s))
        data = denorm(data, mean, std)
        data = seq(data)
        data = renorm(data, mean, std)
    return data


def gaussian_blur(blur, data):
    if blur > 0.5:
        sigma = np.random.uniform(0.15, 1.15)
        kernel_size_y = int(
            np.floor(
                np.ceil(0.1 * data.shape[2]) - 0.5 +
                np.ceil(0.1 * data.shape[2]) % 2))
        kernel_size_x = int(
            np.floor(
                np.ceil(0.1 * data.shape[3]) - 0.5 +
                np.ceil(0.1 * data.shape[3]) % 2))
        kernel_size = (kernel_size_y, kernel_size_x)
        seq = nn.Sequential(
            kornia.filters.GaussianBlur2d(
                kernel_size=kernel_size, sigma=(sigma, sigma)))
        data = seq(data)
    return data


class Masking_2d(nn.Module):
    def __init__(self, block_size, ratio, color_jitter_s, color_jitter_p, blur, mean=None, std=None):
        super(Masking_2d, self).__init__()

        self.block_size = block_size
        self.ratio = ratio

        self.augmentation_params = None
        if (color_jitter_p > 0 and color_jitter_s > 0) or blur:
            print('[Masking] Use color augmentation.')
            self.augmentation_params = {
                'color_jitter': random.uniform(0, 1),
                'color_jitter_s': color_jitter_s,
                'color_jitter_p': color_jitter_p,
                'blur': random.uniform(0, 1) if blur else 0,
                'mean': mean,
                'std': std
            }

    @torch.no_grad()
    def forward(self, img: Tensor):
        img = img.clone()
        B, _, H, W = img.shape

        if self.augmentation_params is not None:
            img = strong_transform(self.augmentation_params, data=img.clone())

        mshape = B, 1, round(H / self.block_size), round(W / self.block_size)
        input_mask = torch.rand(mshape, device=img.device)
        input_mask = (input_mask > self.ratio).float()
        input_mask = resize(input_mask, size=(H, W), mode='bilinear', align_corners=False)
        masked_img = img * input_mask

        return masked_img

class Masking(nn.Module):
    def __init__(self, block_size, ratio, color_jitter_s, color_jitter_p, blur, mean=None, std=None):
        super(Masking, self).__init__()

        self.block_size = block_size
        self.ratio = ratio

        self.augmentation_params = None
        if (color_jitter_p > 0 and color_jitter_s > 0) or blur:
            print('[Masking] Use color augmentation.')
            self.augmentation_params = {
                'color_jitter': random.uniform(0, 1),
                'color_jitter_s': color_jitter_s,
                'color_jitter_p': color_jitter_p,
                'blur': random.uniform(0, 1) if blur else 0,
                'mean': mean,
                'std': std
            }

    @torch.no_grad()
    def forward(self, img: Tensor):
        img = img.clone()
        B, _, H, W, D = img.shape

        if self.augmentation_params is not None:
            img = strong_transform(self.augmentation_params, data=img.clone())

        mshape = B, 1, round(H / self.block_size), round(W / self.block_size), round(D / self.block_size)
        input_mask = torch.rand(mshape, device=img.device)
        input_mask = (input_mask > self.ratio).float()
        input_mask = resize(input_mask, size=(H, W, D), mode='trilinear', align_corners=False)
        masked_img = img * input_mask

        return masked_img
    

class DepthWiseMasking(nn.Module):
    def __init__(self, block_size, ratio, color_jitter_s, color_jitter_p, blur, mean=None, std=None):
        super(DepthWiseMasking, self).__init__()

        self.block_size = block_size
        self.ratio = ratio

        self.augmentation_params = None
        if (color_jitter_p > 0 and color_jitter_s > 0) or blur:
            print('[Masking] Use color augmentation.')
            self.augmentation_params = {
                'color_jitter': random.uniform(0, 1),
                'color_jitter_s': color_jitter_s,
                'color_jitter_p': color_jitter_p,
                'blur': random.uniform(0, 1) if blur else 0,
                'mean': mean,
                'std': std
            }

    @torch.no_grad()
    def forward(self, img: Tensor):
        img = img.clone()
        B, _, H, W, D = img.shape

        if self.augmentation_params is not None:
            img = strong_transform(self.augmentation_params, data=img.clone())

        mshape = B, 1, round(H / self.block_size), round(W / self.block_size)
        for i in range(D):
            img_slice = img[:, :, :, :, i]
            input_mask = torch.rand(mshape, device=img.device)
            input_mask = (input_mask > self.ratio).float()
            input_mask = resize(input_mask, size=(H, W), mode='nearest')
            masked_img = img_slice * input_mask
            img[:, :, :, :, i] = masked_img
            
        return img


class VoxelWiseMasking(nn.Module):
    def __init__(self, block_size, ratio, color_jitter_s, color_jitter_p, blur, mean=None, std=None):
        super(VoxelWiseMasking, self).__init__()

        self.block_size = block_size
        self.ratio = ratio

        self.augmentation_params = None
        if (color_jitter_p > 0 and color_jitter_s > 0) or blur:
            print('[Masking] Use color augmentation.')
            self.augmentation_params = {
                'color_jitter': random.uniform(0, 1),
                'color_jitter_s': color_jitter_s,
                'color_jitter_p': color_jitter_p,
                'blur': random.uniform(0, 1) if blur else 0,
                'mean': mean,
                'std': std
            }

    @torch.no_grad()
    def forward(self, img: Tensor):
        img = img.clone()
        B, _, H, W, D = img.shape

        if self.augmentation_params is not None:
            img = strong_transform(self.augmentation_params, data=img.clone())

        mshape = B, 1, round(H / self.block_size), round(W / self.block_size), round(D / self.block_size)
        input_mask = torch.rand(mshape, device=img.device)
        input_mask = (input_mask > self.ratio).float()
        input_mask = resize(input_mask, size=(H, W, D), mode='nearest')
        masked_img = img * input_mask
        img= masked_img
            
        return img
    

class CenterMasking(nn.Module):
    def __init__(self, block_size, ratio, color_jitter_s, color_jitter_p, blur, mean=None, std=None):
        super(CenterMasking, self).__init__()

        self.block_size = block_size
        self.ratio = ratio

        self.augmentation_params = None
        if (color_jitter_p > 0 and color_jitter_s > 0) or blur:
            print('[Masking] Use color augmentation.')
            self.augmentation_params = {
                'color_jitter': random.uniform(0, 1),
                'color_jitter_s': color_jitter_s,
                'color_jitter_p': color_jitter_p,
                'blur': random.uniform(0, 1) if blur else 0,
                'mean': mean,
                'std': std
            }

    @torch.no_grad()
    def forward(self, img: Tensor):
        img = img.clone()
        B, _, H, W, D = img.shape

        if self.augmentation_params is not None:
            img = strong_transform(self.augmentation_params, data=img.clone())

        # create a center mask with ratio
        input_mask = torch.ones((H, W, D), device=img.device)
        h_min = H * (1 - self.ratio) // 2
        h_max = H - h_min
        w_min = W * (1 - self.ratio) // 2
        w_max = W - w_min
        d_min = D * (1 - self.ratio) // 2
        d_max = D - d_min
        # make values int
        h_min = int(h_min)
        h_max = int(h_max)
        w_min = int(w_min)
        w_max = int(w_max)
        d_min = int(d_min)
        d_max = int(d_max)
        input_mask[h_min:h_max, w_min:w_max, d_min:d_max] = 0
        
        masked_img = img * input_mask
        img= masked_img
            
        return img