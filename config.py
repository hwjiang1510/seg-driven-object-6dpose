import argparse
import numpy as np

# globel param
# dataset setting
img_width = 1920
img_height = 1080
resize_width = 608
resize_height = 608
grid_num = 76
img_channel = 3

data_loader_numworkers = 8

class_num = 2 # background, object
num_keypoints = 8 # 8 bbox corners

conf_thresh = 0.3
bestCnt = 10 # choose best 10 points

#
class_weight = [1.6, 3.57] # CE loss weight
alpha = 1
gamma = 3
beta = 1
modulatingFactor = 1

# path
pose_arch_cfg = './data/arch-FPHA.cfg' # network architecture config
cfg_file = './data/data-FPHA.cfg'

object_model_root = './data/FPHA_Object_models'
object_bbox3d_path = './data/FPHA_bbox.npy'

train_img_path = "./data/train_local/list_image_train.txt"
train_hand_annotation = "./data/train_local/hand_annotation_train.txt"
train_object_annotation = './data/train_local/object_annotation_train.txt'

#val_path = "./data/val_index.txt"
#test_path = "./data/test_index_demo.txt"
#save_path = "./save/result/"
#pretrained_path='./pretrained/unetlstm.

out_dir = './FPHA-Out'

# camera intrinsic
k_FPHA = np.array([[1395.749023, 0.0, 935.732544],
                   [0.0, 1395.749268, 540.681030],
                   [0.0, 0.0, 1.0]])
ex_FPHA = np.array([[0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
                    [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
                    [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
                    [0.0, 0.0, 0.0, 1.0]])
k_FPHA_depth = np.array([[475.065948, 0.0, 315.944855],
                   [0.0, 475.065857, 245.287079],
                   [0.0, 0.0, 1.0]])


def args_setting():
    # Training settings
    parser = argparse.ArgumentParser(description='Seg-driven object 6Dof pose')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, metavar='M',
                        help='SGD weight decay (default: 5e-4)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--cfg_file', type=str, default=cfg_file, metavar='N',
                        help='config file path')
    parser.add_argument('--out_dir', type=str, default=out_dir, metavar='N',
                        help='test result out path')
    args = parser.parse_args()
    return args