'''
This python script is used to visualize the masking operation.
You may change the image path, and the visualized image will be saved to mask_visualization.jpg.
'''
import random
import warnings

import PIL

import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image

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

        mshape = B, 1, round(H / self.block_size), round(W / self.block_size)
        input_mask = torch.rand(mshape, device=img.device)
        input_mask = (input_mask > self.ratio).float()
        input_mask = resize(input_mask, size=(H, W), mode='bilinear', align_corners=False)
        masked_img = img * input_mask

        return masked_img

    @torch.no_grad()
    def forward2(self, img: Tensor):
        img = img.clone()
        B, _, H, W = img.shape

        mshape = B, 1, round(H / self.block_size), round(W / self.block_size)
        input_mask = torch.rand(mshape, device=img.device)
        input_mask = (input_mask > self.ratio).float()
        input_mask = resize(input_mask, size=(H, W))
        masked_img = img * input_mask

        return masked_img

if __name__ == '__main__':
    mask = Masking_2d(block_size=16, ratio=0.3, color_jitter_s=0, color_jitter_p=0, blur=0, mean=None, std=None)
    image = PIL.Image.open('./assets/example.png')
    print("input image size: ", image.size)

    # Define the desired height and width
    desired_height = 224
    desired_width = 224

    # Resize the image using PIL
    resized_image = image.resize((desired_width, desired_height))
    resized_image = resized_image.convert('RGB')
    print("resized image size: ", resized_image.size)

    # Convert the PIL image to a Torch tensor
    transform = transforms.ToTensor()
    tensor_image = transform(resized_image)

    # Reshape the tensor to have the desired dimensions
    reshaped_tensor = tensor_image.view(1, 3, desired_height, desired_width)
    masked_img = mask(reshaped_tensor)
    masked2_img = mask.forward2(reshaped_tensor)
    # save
    save_image(torch.cat([reshaped_tensor, masked_img, masked2_img], dim=0), 'mask_visualization.jpg', nrow=3)
