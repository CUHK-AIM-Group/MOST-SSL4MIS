# --------------------------------------------------------
# MOST Main (train)
# Written by Xinyu Liu
# --------------------------------------------------------
import argparse
import logging
import os
import random
import shutil
import sys
import math

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from skimage.measure import label

from dataloaders.acdc import (
    BaseDataSets,
    TwoStreamBatchSampler,
    WeakStrongAugment_Ours,
)
from networks.net_factory import net_factory
from utils import losses, ramps, val_2d
from dataloaders.masking import Masking_2d
from utils.multi_formation import Synthesizer

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./datasets/acdc', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='ACDC_MOST', help='experiment_name')
parser.add_argument('--model', type=str, default='unet_pure', help='model_name')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=7, help='labeled data', choices=[1, 3, 7])
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument(
    "--conf_thresh",
    type=float,
    default=0.75,
    help="confidence threshold for using pseudo-labels",
)
parser.add_argument('--lr_schedule', type=str, default='cosine', choices=['cosine', 'multistep'])
parser.add_argument('--lr_warmup', type=int, default=0)
# costs
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='6.0', help='magnitude')
parser.add_argument('--s_param', type=int,  default=6, help='multinum of random masks')
# mask
parser.add_argument('--mask_type', type=str,  default='mask', choices=['mask', 'depthwise_mask', 'voxelwise_mask', 'none'])
parser.add_argument('--mask_block_size', default=16, type=int)
parser.add_argument('--mask_ratio', default=0.75, type=float)
parser.add_argument('--mask_color_jitter_s', default=0, type=float) # 0.2
parser.add_argument('--mask_color_jitter_p', default=0, type=float) # 0.2
parser.add_argument('--mask_blur', default=False, type=bool) # True
# others
parser.add_argument('--mf_method', type=str, default='uniform_2d', choices=['uniform_2d', 'multi_2d', 'none'], help='multi-formation method')
parser.add_argument('--mf_factor', type=int, default=2, help='factor of multi-formation')
parser.add_argument('--ada_u_weight', type=float, default=1.0, help='loss weight for unlabeled data')

parser.add_argument("--no_color", default=False, action="store_true", help="no color image")
parser.add_argument("--no_blur", default=False, action="store_true", help="no blur image")
parser.add_argument("--rot", type=int, default=359, help="rotation angle")

args = parser.parse_args()

dice_loss = losses.DiceLoss(n_classes=4)

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def save_net_opt(net, optimizer, path):
    state = {
        'net':net.state_dict(),
        'opt':optimizer.state_dict(),
    }
    torch.save(state, str(path))

def get_ACDC_LargestCC(segmentation):
    class_list = []
    for i in range(1, 4):
        temp_prob = segmentation == i * torch.ones_like(segmentation)
        temp_prob = temp_prob.detach().cpu().numpy()
        labels = label(temp_prob)
        # -- with 'try'
        assert(labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        class_list.append(largestCC * i)
    acdc_largestCC = class_list[0] + class_list[1] + class_list[2]
    return torch.from_numpy(acdc_largestCC).cuda()

def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i] #== c *  torch.ones_like(segmentation[i])
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)          
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)
        
        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)

    return torch.Tensor(batch_list).cuda()
    
def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_ACDC_2DLargestCC(probs)      
    return probs

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 5* args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)

def generate_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x*2/3), int(img_y*2/3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w+patch_x, h:h+patch_y] = 0
    loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

def random_mask(img, shrink_param=3):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    x_split, y_split = int(img_x / shrink_param), int(img_y / shrink_param)
    patch_x, patch_y = int(img_x*2/(3*shrink_param)), int(img_y*2/(3*shrink_param))
    mask = torch.ones(img_x, img_y).cuda()
    for x_s in range(shrink_param):
        for y_s in range(shrink_param):
            w = np.random.randint(x_s*x_split, (x_s+1)*x_split-patch_x)
            h = np.random.randint(y_s*y_split, (y_s+1)*y_split-patch_y)
            mask[w:w+patch_x, h:h+patch_y] = 0
            loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

def contact_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_y = int(img_y *4/9)
    h = np.random.randint(0, img_y-patch_y)
    mask[h:h+patch_y, :] = 0
    loss_mask[:, h:h+patch_y, :] = 0
    return mask.long(), loss_mask.long()

dice_loss = losses.DiceLoss_bcp(n_classes=4)
def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16) 
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)#loss = loss_ce
    return loss_dice, loss_ce

def get_acdc_cp_loss(args, labeled_sub_bs, unlabeled_sub_bs, model, weak_batch, label_batch):
    img_a, img_b = weak_batch[:labeled_sub_bs], weak_batch[labeled_sub_bs:args.labeled_bs]
    uimg_a, uimg_b = weak_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], weak_batch[args.labeled_bs + unlabeled_sub_bs:]
    ulab_a, ulab_b = label_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], label_batch[args.labeled_bs + unlabeled_sub_bs:]
    lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
    with torch.no_grad():
        pre_a = model(uimg_a)
        pre_b = model(uimg_b)
        plab_a = get_ACDC_masks(pre_a, nms=1)
        plab_b = get_ACDC_masks(pre_b, nms=1)
        img_mask, loss_mask = generate_mask(img_a)
        unl_label = ulab_a * img_mask + lab_a * (1 - img_mask)
        l_label = lab_b * img_mask + ulab_b * (1 - img_mask)
    net_input_unl = uimg_a * img_mask + img_a * (1 - img_mask)
    net_input_l = img_b * img_mask + uimg_b * (1 - img_mask)
    out_unl = model(net_input_unl)
    out_l = model(net_input_l)
    unl_dice, unl_ce = mix_loss(out_unl, plab_a, lab_a, loss_mask, u_weight=args.u_weight, unlab=True)
    l_dice, l_ce = mix_loss(out_l, lab_b, plab_b, loss_mask, u_weight=args.u_weight)
    loss_ce = unl_ce + l_ce 
    loss_dice = unl_dice + l_dice
    return unl_label,l_label,net_input_unl,net_input_l,out_unl,out_l,loss_ce,loss_dice

def patients_to_slices(patiens_num):
    ref_dict = {"1": 32, "3": 68, "7": 136,
                "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    return ref_dict[str(patiens_num)]

from torch.distributions import Categorical
def get_comp_loss(weak, strong, bs=args.batch_size):
    """get complementary loss and adaptive sample weight.
    Compares least likely prediction (from strong augment) with argmin of weak augment.

    Args:
        weak (batch): weakly augmented batch
        strong (batch): strongly augmented batch

    Returns:
        comp_loss, as_weight
    """
    il_output = torch.reshape(
        strong,
        (
            bs,
            args.num_classes,
            args.patch_size[0] * args.patch_size[1],
        ),
    )
    # calculate entropy for image-level preds (tensor of length labeled_bs)
    as_weight = 1 - (Categorical(probs=il_output).entropy() / np.log(args.patch_size[0] * args.patch_size[1]))
    # batch level average of entropy
    as_weight = torch.mean(as_weight)
    # complementary loss
    comp_labels = torch.argmin(weak.detach(), dim=1, keepdim=False)
    ce_loss = CrossEntropyLoss()
    comp_loss = as_weight * ce_loss(
        torch.add(torch.negative(strong), 1),
        comp_labels,
    )
    return comp_loss, as_weight

def train(args , snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)
    assert args.labeled_bs % 2 == 0, "labeled_bs should be even number"
    assert (args.batch_size-args.labeled_bs) % 2 == 0, "unlabeled_bs should be even number"
     
    model = net_factory(net_type=args.model, class_num=num_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        num=None,
        transform=transforms.Compose([WeakStrongAugment_Ours(args.patch_size, args)]),
    )
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    db_test = BaseDataSets(base_dir=args.root_path, split="test")
    
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    iter_num = 0
    start_epoch = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iter_num = int(iter_num)
    
    multi_formation_synthesizer = Synthesizer(factor=args.mf_factor, decode_type=args.mf_method, patch_size=args.patch_size, labeled_bs=args.labeled_bs, unlabeled_bs=args.batch_size-args.labeled_bs)
    
    if args.mask_type == 'mask':
        masking = Masking_2d(
            block_size=args.mask_block_size,
            ratio=args.mask_ratio,
            color_jitter_s=args.mask_color_jitter_s,
            color_jitter_p=args.mask_color_jitter_p,
            blur=args.mask_blur,)
    elif args.mask_type == 'none':
        masking = lambda x: x
    else:
        raise NotImplementedError
    
    lr_ = base_lr
    for epoch_num in range(start_epoch, max_epoch):
        for i_batch, sampled_batch in enumerate(trainloader):
            weak_batch, strong_batch, label_batch = (
                sampled_batch["image_weak"],
                sampled_batch["image_strong"],
                sampled_batch["label_aug"],
            )
            weak_batch, strong_batch, label_batch = (
                weak_batch.cuda(),
                strong_batch.cuda(),
                label_batch.cuda(),
            )
            label_batch[args.labeled_bs:] = torch.zeros_like(label_batch[args.labeled_bs:])

            mf_weak_batch_u, mf_strong_batch_u \
                = multi_formation_synthesizer.run(weak_batch, strong_batch, label_batch)            
            outputs_weak, outputs_weak_mf_u \
                = model(torch.cat([weak_batch, mf_weak_batch_u], dim=0)).split([weak_batch.shape[0], mf_weak_batch_u.shape[0]], dim=0)
            outputs_weak_soft, outputs_weak_soft_mf_u \
                = torch.softmax(outputs_weak, dim=1), torch.softmax(outputs_weak_mf_u, dim=1)
            logits_u_aug, label_u_aug = torch.max(outputs_weak_soft, dim=1)
            logits_u_aug_mf, label_u_aug_mf = torch.max(outputs_weak_soft_mf_u, dim=1)
            strong_batch = masking(strong_batch)
            mf_strong_batch_u = masking(mf_strong_batch_u)
            outputs_strong, outputs_strong_mf_u \
                = model(torch.cat([strong_batch, mf_strong_batch_u], dim=0)).split([strong_batch.shape[0], mf_strong_batch_u.shape[0]], dim=0)
            outputs_strong_soft, outputs_strong_soft_mf_u \
                = torch.softmax(outputs_strong, dim=1), torch.softmax(outputs_strong_mf_u, dim=1)

            consistency_weight = get_current_consistency_weight(iter_num // 150)

            sup_loss = ce_loss(outputs_weak[: args.labeled_bs], label_batch[:][: args.labeled_bs].long(),) + dice_loss(outputs_weak_soft[: args.labeled_bs], label_batch[: args.labeled_bs].unsqueeze(1))

            comp_loss, as_weight = get_comp_loss(weak=outputs_weak_soft, strong=outputs_strong_soft)

            unsup_loss, pseduo_high_ratio = losses.compute_unsupervised_loss_by_threshold_2d(
                outputs_strong[args.labeled_bs :],
                label_u_aug[args.labeled_bs :],
                logits_u_aug[args.labeled_bs :],
                thresh=args.conf_thresh,
            )
            unsup_loss_mf, pseduo_high_ratio_mf = losses.compute_unsupervised_loss_by_threshold_2d(
                outputs_strong_mf_u,
                label_u_aug_mf,
                logits_u_aug_mf,
                thresh=args.conf_thresh,
            )
            consistency_weight = get_current_consistency_weight(iter_num//150)
            
            loss = sup_loss + args.ada_u_weight * unsup_loss + args.ada_u_weight * unsup_loss_mf + as_weight * comp_loss
            
            unl_label, l_label, net_input_unl, net_input_l, out_unl, out_l, loss_ce, loss_dice = get_acdc_cp_loss(args, labeled_sub_bs, unlabeled_sub_bs, model, weak_batch, label_batch)

            loss += (loss_ce + loss_dice) /2        

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1

            if args.lr_schedule == 'multistep':
                if iter_num % 2500 == 0:
                    lr_ = base_lr * 0.1 ** (iter_num // 2500)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
            elif args.lr_schedule == 'cosine':
                warmup_iterations = 500 # Default to 500.
                if iter_num >= warmup_iterations:
                    lr_ = base_lr * (1 + math.cos(math.pi * (iter_num - warmup_iterations) / (max_iterations - warmup_iterations))) / 2
                else:
                    warmup_factor = iter_num / warmup_iterations
                    lr_ = base_lr * warmup_factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/sup_loss', sup_loss, iter_num)
            writer.add_scalar('info/unsup_loss', unsup_loss, iter_num)
            writer.add_scalar('info/comp_loss', as_weight * comp_loss, iter_num)
            writer.add_scalar('info/unsuploss_weight', consistency_weight * unsup_loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('info/lr', lr_, iter_num)     

            logging.info('iter %d : loss : %03f, loss_sup: %03f, loss_unsup: %03f, loss_unsup_mf: %03f, loss_comp: %03f, asw*loss_comp: %03f, mix_ce: %03f, mix_dice: %03f, high_ratio: %03f, high_ratio_mf: %03f'
                        % (iter_num, loss.item(), sup_loss.item(), unsup_loss.item(), unsup_loss_mf.item(), comp_loss.item(), as_weight.item() * comp_loss.item(),
                           loss_ce.item(), loss_dice.item(), pseduo_high_ratio.item(), pseduo_high_ratio_mf.item()))
                
            if iter_num % 20 == 0:
                image = net_input_unl[1, 0:1, :, :]
                writer.add_image('train/Un_Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(out_unl, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Un_Prediction', outputs[1, ...] * 50, iter_num)
                labs = unl_label[1, ...].unsqueeze(0) * 50
                writer.add_image('train/Un_GroundTruth', labs, iter_num)

                image_l = net_input_l[1, 0:1, :, :]
                writer.add_image('train/L_Image', image_l, iter_num)
                outputs_l = torch.argmax(torch.softmax(out_l, dim=1), dim=1, keepdim=True)
                writer.add_image('train/L_Prediction', outputs_l[1, ...] * 50, iter_num)
                labs_l = l_label[1, ...].unsqueeze(0) * 50
                writer.add_image('train/L_GroundTruth', labs_l, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes) # ACDC dataset
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                assert len(metric_list) == num_classes - 1
                for class_i in range(num_classes - 1):
                    logging.info(
                        "iteration %d: val_cls%d_dice : %f val_cls%d_hd95 : %f val_cls%d_jaccard : %f, val_cls%d_asd : %f"
                        % (iter_num, class_i + 1, metric_list[class_i, 0], class_i + 1, metric_list[class_i, 1], class_i + 1, metric_list[class_i, 2], class_i + 1, metric_list[class_i, 3])
                    )

                performance = np.mean(metric_list, axis=0)[0]
                mean_jaccard = np.mean(metric_list, axis=0)[1]
                mean_hd95 = np.mean(metric_list, axis=0)[2]
                mean_asd = np.mean(metric_list, axis=0)[3]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_jaccard', mean_jaccard, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)
                writer.add_scalar('info/val_mean_asd', mean_asd, iter_num)

                logging.info('iteration %d : mean_dice : %f, mean_hd95 : %f, mean_jaccard : %f, mean_asd : %f' % (iter_num, performance, mean_jaccard, mean_hd95, mean_asd))
                if performance > best_performance:
                    best_performance = performance
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_best_path)
                    # Only test the performance on testloader for the best model selected by validation set
                    test_metric_list = 0.0
                    for _, sampled_batch in enumerate(testloader):
                        metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                        test_metric_list += np.array(metric_i)
                    test_metric_list = test_metric_list / len(db_test)
                    for class_i in range(num_classes - 1):
                        logging.info(
                            "iteration %d: test_cls%d_dice : %f test_cls%d_hd95 : %f test_cls%d_jaccard : %f, test_cls%d_asd : %f"
                            % (iter_num, class_i + 1, metric_list[class_i, 0], class_i + 1, metric_list[class_i, 1], class_i + 1, metric_list[class_i, 2], class_i + 1, metric_list[class_i, 3])
                        )
                    test_performance = np.mean(test_metric_list, axis=0)[0]
                    test_mean_jaccard = np.mean(test_metric_list, axis=0)[1]
                    test_mean_hd95 = np.mean(test_metric_list, axis=0)[2]
                    test_mean_asd = np.mean(test_metric_list, axis=0)[3]
                    logging.info('iteration %d : test_mean_dice : %f, test_mean_hd95 : %f, test_mean_jaccard : %f, test_mean_asd : %f' % (iter_num, test_performance, test_mean_jaccard, test_mean_hd95, test_mean_asd))

                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()



if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # -- path to save models
    self_snapshot_path = "./ACDC_{}_{}_bs{}_labbs{}_{}labeled_mratio{}".format(args.exp, args.model, args.batch_size, args.labeled_bs, args.labelnum, args.mask_ratio)
    for snapshot_path in [self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
    print("snapshot_path: ", self_snapshot_path)
    shutil.copy('./code/train_ACDC_MOST.py', self_snapshot_path)

    logging.basicConfig(filename=self_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, self_snapshot_path)
