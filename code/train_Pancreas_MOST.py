import os
import sys
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import random
import math
import numpy as np
import torch
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from utils import losses, ramps, test_3d_patch
from dataloaders.datasets_3d import StrongWeakPancreas
from dataloaders.masking import Masking, DepthWiseMasking, VoxelWiseMasking
from networks.net_factory import net_factory
from utils.BCP_utils import context_mask, mix_loss, parameter_sharing
from utils.multi_formation import Synthesizer

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./datasets/pancreas', help='Name of Dataset')
parser.add_argument('--exp', type=str,  default='Pancreas_MOST', help='exp_name')
parser.add_argument('--model', type=str, default='VNet_pure', help='model_name')
parser.add_argument('--max_iteration', type=int,  default=15000, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int,  default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=12, help='trained samples')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='7', help='GPU to use')
parser.add_argument("--ema_decay", type=float, default=0.99, help="ema_decay")
parser.add_argument('--consistency', type=float, default=1, help='consistency')
parser.add_argument(
    "--conf_thresh",
    type=float,
    default=0.75,
    help="confidence threshold for using pseudo-labels",
)
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='10.0', help='magnitude')
# mask
parser.add_argument('--mask_type', type=str,  default='mask', choices=['mask', 'depthwise_mask', 'voxelwise_mask', 'none'])
parser.add_argument('--mask_block_size', default=16, type=int)
parser.add_argument('--mask_ratio', default=0.75, type=float)
parser.add_argument('--mask_color_jitter_s', default=0, type=float) # 0.2
parser.add_argument('--mask_color_jitter_p', default=0, type=float) # 0.2
parser.add_argument('--mask_blur', default=False, type=bool) # True
parser.add_argument('--ema', action='store_true') # Whether to use EMA model
parser.add_argument('--no_strong_gamma', action='store_true')
parser.add_argument('--no_strong_noise', default=True, type=bool, help='Whether to use Gaussian noise as strong augmentation. Default: False')
# optimizer
parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam', 'adamw'])
parser.add_argument('--weight_decay', type=float, default=0.0001)
# lr schedule
parser.add_argument('--lr_schedule', type=str, default='cosine', choices=['cosine', 'multistep'])
parser.add_argument('--lr_warmup', type=int, default=0)
# others
parser.add_argument('--ipe', type=int,  default=150, help='iters per epoch (for adjusting consistency weights, the larger, the smaller weights)')
parser.add_argument('--bcp_mask_ratio', default=2/3, type=float)
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--bcp_unsup_weight', type=int,  default=0, help='multiply consistency weight for bcp_loss_u')
# others
parser.add_argument('--mf_method', type=str, default='uniform', choices=['uniform', 'multi', 'none'], help='multi-formation method')
parser.add_argument('--mf_factor', type=int, default=2, help='factor of multi-formation')
parser.add_argument('--ada_u_weight', type=float, default=1.0, help='loss weight for unlabeled data')
args = parser.parse_args()

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

train_data_path = args.root_path
snapshot_path = "./Pancreas_{}_{}_bs{}_labbs{}_{}labeled_mratio{}".format(args.exp, args.model, args.batch_size, args.labeled_bs, args.labelnum, args.mask_ratio)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ["OMP_NUM_THREADS"] = "8"
max_iterations = args.max_iteration
base_lr = args.base_lr

# if args.deterministic:
cudnn.benchmark = False
cudnn.deterministic = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

patch_size = (96, 96, 96)
num_classes = 2

def normalize(tensor):
    min_val = tensor.min(1, keepdim=True)[0]
    max_val = tensor.max(1, keepdim=True)[0]
    result = tensor - min_val
    result = result / max_val
    return result

def get_comp_loss(weak, strong):
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
            args.batch_size,
            num_classes,
            patch_size[0] * patch_size[1] * patch_size[2],
        ),
    )
    # calculate entropy for image-level preds (tensor of length labeled_bs)
    as_weight = 1 - (Categorical(probs=il_output).entropy() / np.log(patch_size[0] * patch_size[1] * patch_size[2]))
    # batch level average of entropy
    as_weight = torch.mean(as_weight)
    # complementary loss
    comp_labels = torch.argmin(weak.detach(), dim=1, keepdim=False)
    comp_loss = as_weight * ce_loss(
        torch.add(torch.negative(strong), 1),
        comp_labels,
    )
    return comp_loss, as_weight
    
def get_cut_mask(out, thres=0.5, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks

from skimage.measure import label
def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)
    
    return torch.Tensor(batch_list).cuda()

def update_ema_variables(model, ema_model, alpha, global_step):
    # teacher network: ema_model
    # student network: model
    # Use the true average until the exponential average is more correct
    alpha = 0.99
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('./code/', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("Arguments:")
    for arg in vars(args):
        logging.info("\t{}: {}".format(arg, getattr(args, arg)))
    logging.info("\n")

    sub_bs = int(args.labeled_bs/2)
    def create_model(ema=False):
        model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="test")
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")

    mask_strategy = {'mask': Masking, 'depthwise_mask': DepthWiseMasking, 'voxelwise_mask': VoxelWiseMasking}
    masking = mask_strategy[args.mask_type](
        block_size=args.mask_block_size,
        ratio=args.mask_ratio,
        color_jitter_s=args.mask_color_jitter_s,
        color_jitter_p=args.mask_color_jitter_p,
        blur=args.mask_blur,
    ) if args.mask_type != 'none' else lambda x: x
    labelnum = args.labelnum
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
        
    def create_dataloader():
        if labelnum == 12:
            train_labset =StrongWeakPancreas(args.root_path, split='lab12')
            train_unlabset = StrongWeakPancreas(args.root_path, split='unlab12')
        testset = StrongWeakPancreas(args.root_path, split='test') 
        
        trainlab_loader = DataLoader(train_labset, batch_size=args.labeled_bs, shuffle=True, num_workers=0)
        trainunlab_loader = DataLoader(train_unlabset, batch_size=args.batch_size-args.labeled_bs, shuffle=True, num_workers=0)
        test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

        logging.info("{} batches for lab per epoch.".format(len(trainlab_loader)))
        logging.info("{} batches for unlab per epoch.".format(len(trainunlab_loader)))
        logging.info("{} samples for test.\n".format(len(test_loader)))
        return trainlab_loader, trainunlab_loader, test_loader

    trainlab_loader, trainunlab_loader, test_loader = create_dataloader()

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(2)
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
    
    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainlab_loader)))
    iter_num = 0
    best_dice = 0
    max_epoch = max_iterations // len(trainlab_loader) + 1
    lr_ = base_lr
    iterator = range(max_epoch)
    multi_formation_synthesizer = Synthesizer(factor=args.mf_factor, decode_type=args.mf_method, patch_size=patch_size, labeled_bs=args.labeled_bs, unlabeled_bs=args.batch_size-args.labeled_bs)
    
    for epoch_num in iterator:
        for step, (labeled_batch, unlabeled_batch) in enumerate(zip(trainlab_loader,trainunlab_loader)):
            weak_batch_l, label_l = labeled_batch['image_weak'].cuda(), labeled_batch['label'].cuda()
            weak_batch_u, strong_batch_u = unlabeled_batch['image_weak'].cuda(), unlabeled_batch['image_strong'].cuda()

            weak_batch = torch.cat([weak_batch_l, weak_batch_u], dim=0)
            strong_batch = torch.cat([torch.zeros_like(strong_batch_u), strong_batch_u], dim=0)
            label_batch = torch.cat([label_l, torch.zeros_like(label_l)], dim=0)

            weak_batch, strong_batch, label_batch = (
                weak_batch.cuda(),
                strong_batch.cuda(),
                label_batch.cuda(),
            )
            
            label_batch[args.labeled_bs :] = 0

            model.train()
            
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

            consistency_weight = get_current_consistency_weight(iter_num // args.ipe)

            sup_loss = ce_loss(outputs_weak[: args.labeled_bs], label_batch[:][: args.labeled_bs].long(),) + dice_loss(outputs_weak_soft[: args.labeled_bs], label_batch[: args.labeled_bs].unsqueeze(1))

            comp_loss, as_weight = get_comp_loss(weak=outputs_weak_soft, strong=outputs_strong_soft)

            unsup_loss, pseduo_high_ratio = losses.compute_unsupervised_loss_by_threshold(
                outputs_strong[args.labeled_bs :],
                label_u_aug[args.labeled_bs :],
                logits_u_aug[args.labeled_bs :],
                thresh=args.conf_thresh,
            )
            unsup_loss_mf, pseduo_high_ratio_mf = losses.compute_unsupervised_loss_by_threshold(
                outputs_strong_mf_u,
                label_u_aug_mf,
                logits_u_aug_mf,
                thresh=args.conf_thresh,
            )
            n_mix = args.labeled_bs // 2
            img_a, img_b = weak_batch[:n_mix], weak_batch[n_mix:2*n_mix]
            lab_a, lab_b = label_batch[:n_mix], label_batch[n_mix:2*n_mix]
            unimg_a, unimg_b = weak_batch[args.labeled_bs:args.labeled_bs+n_mix], weak_batch[args.labeled_bs+n_mix:args.labeled_bs+2*n_mix]
            with torch.no_grad():
                unoutput_a = model(unimg_a)
                unoutput_b = model(unimg_b)
                plab_a = get_cut_mask(unoutput_a, nms=1)
                plab_b = get_cut_mask(unoutput_b, nms=1)
                img_mask, loss_mask = context_mask(img_a, args.bcp_mask_ratio)

            mixl_img = img_a * img_mask + unimg_a * (1 - img_mask)
            mixu_img = unimg_b * img_mask + img_b * (1 - img_mask)
            mixl_lab = lab_a * img_mask + plab_a * (1 - img_mask)
            mixu_lab = plab_b * img_mask + lab_b * (1 - img_mask)
            outputs_l = model(mixl_img)
            outputs_u = model(mixu_img)
            loss_l = mix_loss(outputs_l, lab_a, plab_a, loss_mask, u_weight=args.u_weight)
            loss_u = mix_loss(outputs_u, plab_b, lab_b, loss_mask, u_weight=args.u_weight, unlab=True)
            if args.bcp_unsup_weight:
                loss_u = loss_u * consistency_weight

            loss = sup_loss + args.ada_u_weight * unsup_loss + args.ada_u_weight * unsup_loss_mf + as_weight * comp_loss + loss_l + loss_u

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            iter_num = iter_num + 1

            writer.add_scalar('1_Loss/sup_loss', sup_loss, iter_num)
            writer.add_scalar('1_Loss/unsup_loss', unsup_loss, iter_num)
            writer.add_scalar('1_Loss/comp_loss', as_weight * comp_loss, iter_num)
            writer.add_scalar('1_Loss/unsuploss_weight', consistency_weight * unsup_loss, iter_num)
            writer.add_scalar('1_Loss/total_loss', loss, iter_num)
            writer.add_scalar('1_Loss/bcp_loss_l', loss_l, iter_num)
            writer.add_scalar('1_Loss/bcp_loss_u', loss_u, iter_num)
                        
            logging.info('iteration %d : loss : %03f, loss_sup: %03f, loss_unsup: %03f, loss_unsup_mf: %03f, loss_comp: %03f, asw*loss_comp: %03f, bcp_loss_l: %03f, bcp_loss_u: %03f, high_ratio: %03f, high_ratio_mf: %03f'
                        % (iter_num, loss.item(), sup_loss.item(), unsup_loss.item(), unsup_loss_mf.item(), comp_loss.item(), as_weight.item() * comp_loss.item(),
                           loss_l.item(), loss_u.item(), pseduo_high_ratio.item(), pseduo_high_ratio_mf.item()))
            writer.add_scalar('3_consist_weight', consistency_weight, iter_num)

            if iter_num >= 800 and iter_num % 200 == 0:
                ins_width = 2
                B,C,H,W,D = outputs_weak.size()
                snapshot_img = torch.zeros(size = (D, 3, 3*H + 3 * ins_width, W + ins_width), dtype = torch.float32)

                snapshot_img[:,:, H:H+ ins_width,:] = 1
                snapshot_img[:,:, 2*H + ins_width:2*H + 2*ins_width,:] = 1
                snapshot_img[:,:, 3*H + 2*ins_width:3*H + 3*ins_width,:] = 1
                snapshot_img[:,:, :,W:W+ins_width] = 1

                seg_out = outputs_weak_soft[args.labeled_bs,1,...].permute(2,0,1) # y
                target =  label_batch[args.labeled_bs,...].permute(2,0,1)
                train_img = weak_batch[args.labeled_bs,0,...].permute(2,0,1)

                snapshot_img[:, 0,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 1,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 2,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))

                snapshot_img[:, 0, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 1, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 2, H+ ins_width:2*H+ ins_width,:W] = target

                snapshot_img[:, 0, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                snapshot_img[:, 1, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                snapshot_img[:, 2, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                
                writer.add_images('Epoch_%d_Iter_%d_unlabel'% (epoch_num, iter_num), snapshot_img)

                seg_out = outputs_weak_soft[0,1,...].permute(2,0,1) # y
                target =  label_batch[0,...].permute(2,0,1)
                train_img = weak_batch[0,0,...].permute(2,0,1)

                snapshot_img[:, 0,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 1,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 2,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))

                snapshot_img[:, 0, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 1, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 2, H+ ins_width:2*H+ ins_width,:W] = target

                snapshot_img[:, 0, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                snapshot_img[:, 1, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                snapshot_img[:, 2, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out

                writer.add_images('Epoch_%d_Iter_%d_label'% (epoch_num, iter_num), snapshot_img)
            
            # change lr
            if args.lr_schedule == 'multistep':
                if iter_num % 2500 == 0:
                    lr_ = base_lr * 0.1 ** (iter_num // 2500)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
            elif args.lr_schedule == 'cosine':
                warmup_iterations = 500 # Hard code it to 500 now. May change it later.
                if iter_num >= warmup_iterations:
                    lr_ = base_lr * (1 + math.cos(math.pi * (iter_num - warmup_iterations) / (max_iterations - warmup_iterations))) / 2
                else:
                    warmup_factor = iter_num / warmup_iterations
                    lr_ = base_lr * warmup_factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_


            if iter_num % 200 == 0:
                model.eval()
                dice_sample = test_3d_patch.var_all_case_Pancreas(model, num_classes=num_classes, patch_size=patch_size, stride_xy=16, stride_z=4, data_path=args.root_path)
                logging.info("Dice score at {}-th iteration is {}".format(iter_num, round(dice_sample, 4)))
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    logging.info("Dice score of best model is {}".format(best_dice))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_best_path))
                writer.add_scalar('4_Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', best_dice, iter_num)
                model.train()


            if iter_num >= max_iterations:
                break

        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
