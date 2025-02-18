# --------------------------------------------------------
# MOST Main (test)
# Written by Xinyu Liu
# --------------------------------------------------------
import os
import argparse
import torch

from networks.net_factory import net_factory
from utils.test_3d_patch import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./datasets', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default=None, help='exp_name')
parser.add_argument('--model', type=str,  default='VNet_pure', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--detail', type=int,  default=1, help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=1, help='apply NMS post-procssing?')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=4, help='labeled data')
parser.add_argument('--ema', action='store_true') # Whether to use EMA model

FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
os.environ["OMP_NUM_THREADS"] = "8"
num_classes = 2

assert FLAGS.exp is not None
test_save_path = FLAGS.exp + "/{}_predictions/".format(FLAGS.model)

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)
FLAGS.root_path = os.path.join(FLAGS.root_path, 'pancreas') # for LA dataset
with open(FLAGS.root_path + '/test.list', 'r') as f:
    image_list = f.readlines()
image_list = [os.path.join(FLAGS.root_path, "data", item.replace('\n', '') + ".h5") for item in image_list]


def test_calculate_metric():
    model = net_factory(net_type=FLAGS.model, in_chns=1, class_num=num_classes, mode="test")
    
    save_model_path = os.path.join(FLAGS.exp, '{}_best_model.pth'.format(FLAGS.model))
    model.load_state_dict(torch.load(save_model_path))
    print("init weight from {}".format(save_model_path))
    model.eval()

    avg_metric = test_all_case(model, image_list, num_classes=num_classes,
                           patch_size=(96, 96, 96), stride_xy=16, stride_z=4, 
                           save_result=True, test_save_path=test_save_path,
                           metric_detail=FLAGS.detail, nms=FLAGS.nms)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()
    print(metric)
