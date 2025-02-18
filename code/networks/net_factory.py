from networks.unet_pure import UNet_pure
from networks.VNet_pure import VNet_pure

def net_factory(net_type="unet", in_chns=1, class_num=2, mode = "train"):
    if net_type == "unet_pure":
        net = UNet_pure(in_chns=in_chns, class_num=class_num).cuda()
    if net_type == "VNet_pure" and mode == "train":
        net = VNet_pure(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    if net_type == "VNet_pure" and mode == "test":
        net = VNet_pure(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    return net
