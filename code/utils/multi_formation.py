import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

class Synthesizer():
    """Condensed data class
    """
    def __init__(self, factor, decode_type, patch_size, labeled_bs, unlabeled_bs, device='cuda', unsup_only=True, max_formation=8):
        self.factor = max(1, factor)
        self.decode_type = decode_type
        if len(patch_size) == 2:
            self.resize = nn.Upsample(size=patch_size, mode='bilinear')
            self.resize_mask = nn.Upsample(size=patch_size)
        elif len(patch_size) == 3:
            self.resize = nn.Upsample(size=patch_size, mode='trilinear')
            self.resize_mask = nn.Upsample(size=patch_size)
        else:
            raise NotImplementedError(f"Unknown patch size: {patch_size}")
        self.labeled_bs = labeled_bs
        self.unlabeled_bs = unlabeled_bs
        self.device = device
        self.unsup_only = unsup_only
        print(f"Factor: {self.factor} ({self.decode_type})")
        self.max_formation = max_formation
        self.max_formation_multi = 0

        # Raise a warning if the decode type is ''multi' but the factor is 2, which is the same as 'uniform'
        if self.decode_type == 'multi' and self.factor == 2:
            print("Warning: decode type is 'multi' but factor is 2, which is the same as 'uniform'")

        if not self.unsup_only:
            print("Warning: labeled data is also used for multi formation. This feature is not fully tested and may not work properly.")

    def decode_zoom_uniform_3d(self, img_weak, img_strong, target, factor):
        """Uniform multi-formation
        """
        max_formation = self.max_formation if self.max_formation_multi == 0 else self.max_formation_multi
        assert img_weak.shape == img_strong.shape, "Weak and strong images should have the same shape, but got {} and {}".format(img_weak.shape, img_strong.shape)
        target = torch.unsqueeze(target, 1)

        n, c, h, w, d = img_weak.shape
        assert h == w, "Only support cubic with the same height and width, but got {} and {}".format(h, w)
        remained = h % factor
        assert remained == 0, "Image size should be divisible by factor, but got h={}, factor={}".format(h, factor)
        s_crop = ceil(h / factor)

        if not self.unsup_only:
            cropped_weak_l = []
            cropped_target_l = []
            img_weak_l = img_weak[:self.labeled_bs]
            target_l = target[:self.labeled_bs]

        cropped_weak_u = []
        cropped_strong_u = []
        img_weak_u = img_weak[self.labeled_bs:]
        img_strong_u = img_strong[self.labeled_bs:]

        for i in range(factor):
            for j in range(factor):
                h_loc = i * s_crop
                w_loc = j * s_crop
                
                if not self.unsup_only:
                    cropped_weak_l.append(img_weak_l[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop, :])
                    cropped_target_l.append(target_l[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop, :])
                cropped_weak_u.append(img_weak_u[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop, :])
                cropped_strong_u.append(img_strong_u[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop, :])
                
        if not self.unsup_only:
            cropped_weak_l = torch.cat(cropped_weak_l).to(img_weak.device)
            cropped_target_l = torch.cat(cropped_target_l).to(img_weak.device)
            
        if len(cropped_weak_u) > max_formation: # random select max_formation formations if the number of formations is larger than max_formation
            idx = np.random.choice(len(cropped_weak_u), max_formation, replace=False)
            cropped_weak_u = [cropped_weak_u[i] for i in idx]
            cropped_strong_u = [cropped_strong_u[i] for i in idx]
        cropped_weak_u = torch.cat(cropped_weak_u).to(img_weak.device)
        cropped_strong_u = torch.cat(cropped_strong_u).to(img_weak.device)
        
        if not self.unsup_only:
            weak_dec_l = self.resize(cropped_weak_l)
            target_dec_l = torch.squeeze(self.resize_mask(cropped_target_l.float()), 1).long()
        weak_dec_u = self.resize(cropped_weak_u)
        strong_dec_u = self.resize(cropped_strong_u)
        del cropped_weak_u, cropped_strong_u # save memory

        if not self.unsup_only:
            return weak_dec_l, target_dec_l, weak_dec_u, strong_dec_u
        else:
            return weak_dec_u, strong_dec_u

    def decode_zoom_uniform_2d(self, img_weak, img_strong, target, factor):
        """Uniform multi-formation
        """
        max_formation = self.max_formation
        assert img_weak.shape == img_strong.shape, "Weak and strong images should have the same shape"
        target = torch.unsqueeze(target, 1)

        n, c, h, w = img_weak.shape
        assert h == w, "Only support square with the same height and width, but got {} and {}".format(h, w)
        remained = h % factor
        assert remained == 0, "Image size should be divisible by factor"
        s_crop = ceil(h / factor)

        if not self.unsup_only:
            cropped_weak_l = []
            cropped_target_l = []
            img_weak_l = img_weak[:self.labeled_bs]
            target_l = target[:self.labeled_bs]

        cropped_weak_u = []
        cropped_strong_u = []
        img_weak_u = img_weak[self.labeled_bs:]
        img_strong_u = img_strong[self.labeled_bs:]

        for i in range(factor):
            for j in range(factor):
                h_loc = i * s_crop
                w_loc = j * s_crop
                
                if not self.unsup_only:
                    cropped_weak_l.append(img_weak_l[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop])
                    cropped_target_l.append(target_l[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop])
                cropped_weak_u.append(img_weak_u[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop])
                cropped_strong_u.append(img_strong_u[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop])
                
        if not self.unsup_only:
            cropped_weak_l = torch.cat(cropped_weak_l).to(img_weak.device)
            cropped_target_l = torch.cat(cropped_target_l).to(img_weak.device)
        # random select max_formation formations if the number of formations is larger than max_formation
        if len(cropped_weak_u) > max_formation:
            idx = np.random.choice(len(cropped_weak_u), max_formation, replace=False)
            cropped_weak_u = [cropped_weak_u[i] for i in idx]
            cropped_strong_u = [cropped_strong_u[i] for i in idx]
        cropped_weak_u = torch.cat(cropped_weak_u).to(img_weak.device)
        cropped_strong_u = torch.cat(cropped_strong_u).to(img_weak.device)
        
        if not self.unsup_only:
            weak_dec_l = self.resize(cropped_weak_l)
            target_dec_l = torch.squeeze(self.resize_mask(cropped_target_l.float()), 1).long()
        weak_dec_u = self.resize(cropped_weak_u)
        strong_dec_u = self.resize(cropped_strong_u)
        del cropped_weak_u, cropped_strong_u

        if not self.unsup_only:
            return weak_dec_l, target_dec_l, weak_dec_u, strong_dec_u
        else:
            return weak_dec_u, strong_dec_u

    def run(self, img_weak, img_strong, target):
        if self.decode_type == 'uniform':
            return self.decode_zoom_uniform_3d(img_weak, img_strong, target, self.factor)
        elif self.decode_type == 'uniform_2d':
            return self.decode_zoom_uniform_2d(img_weak, img_strong, target, self.factor)
        elif self.decode_type == 'none':
            return img_weak, target
        else:
            raise NotImplementedError(f"Unknown decode type: {self.decode_type}")
