from torch.utils.data import Dataset
from PIL import Image
import torch
import config
import torchvision.transforms as transforms
import numpy as np
from sklearn import preprocessing
from utils import *

def readTxt(file_path):
    img_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            item = lines.strip()
            img_list.append(item)
    file_to_read.close()
    return img_list

def str_to_np(line):
    d = line.split(' ')
    d = [float(it) for it in d]
    return np.array(d).reshape((4,4)).T

def prepare_seg_label(pose, vertex, intrinsics, extrinsic):
    width, height = config.img_width, config.img_height
    maskImg = np.zeros((height, width), np.uint8)
    vp = vertices_reprojection_FPHA(vertex, pose, intrinsics, extrinsic)
    for p in vp:
        if p[0] != p[0] or p[1] != p[1]:  # check nan
            continue
        maskImg = cv2.circle(maskImg, (int(p[0]), int(p[1])), 1, 1, -1) # radius=1, color=1, thickness=-1
    kernel = np.ones((5, 5), np.uint8)
    maskImg = cv2.morphologyEx(maskImg, cv2.MORPH_CLOSE, kernel)
    mask_binary = np.ma.getmaskarray(np.ma.masked_not_equal(maskImg, 0)).astype(int)
    return maskImg, mask_binary

def prepare_reg_label(pose, bbox_3d, intrinsics, extrinsic):
    return vertices_reprojection_FPHA(bbox_3d, pose, intrinsics, extrinsic)

def draw_reg_label(vp, width, height):
    maskImg = np.zeros((height, width), np.uint8)
    for p in vp:
        if p[0] != p[0] or p[1] != p[1]:  # check nan
            continue
        maskImg = cv2.circle(maskImg, (int(p[0]), int(p[1])), 5, 255, -1) # radius=1, color=1, thickness=-1
    return maskImg

def align_reg_label(bbox2d, cord_upleft):
    # 0 for x(width), 1 for y(height)
    reg_label = bbox2d.copy()
    #print(cord_upleft)
    reg_label[:,0] = reg_label[:,0] - cord_upleft[0]
    reg_label[:,1] = reg_label[:,1] - cord_upleft[1]
    return reg_label

def reshape_reg_label(bbox2d, cur_size, target_size):
    # 0 for x(width), 1 for y(height)
    reg_label = bbox2d.copy()
    reg_label[:, 0] = reg_label[:, 0] * target_size[0] / cur_size[0]
    reg_label[:, 1] = reg_label[:, 1] * target_size[1] / cur_size[1]
    return reg_label

def image_label_crop(img, seg_label, front_mask):
    min_y, max_y = np.min(np.where(front_mask == 1)[0]), np.max(np.where(front_mask == 1)[0])
    min_x, max_x = np.min(np.where(front_mask == 1)[1]), np.max(np.where(front_mask == 1)[1])
    mean_y, mean_x = (min_y + max_y) // 2, (min_x + max_x) // 2
    width, height = max_x - min_x, max_y - min_y

    crop_xmin, crop_xmax = int(mean_x - 2 / 2 * (width)), int(mean_x + 2 / 2 * (width))
    crop_ymin, crop_ymax = int(mean_y - 2 / 2 * (height)), int(mean_y + 2 / 2 * (height))
    crop_xmin = max(0, crop_xmin)
    crop_xmax = min(config.img_width - 1, crop_xmax)
    crop_ymin = max(0, crop_ymin)
    crop_ymax = min(config.img_height - 1, crop_ymax)

    data_crop = img.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
    seg_label_crop = seg_label[crop_ymin:crop_ymax, crop_xmin: crop_xmax]
    return data_crop, seg_label_crop, [crop_xmin, crop_ymin]

class FPHA_hand(Dataset):
    def __init__(self, file_path, hand_label, object_label, transforms, mesh, bbox_3d, intrinsics, extrinsics):
        self.img_list = readTxt(file_path)
        self.hand_label_list = readTxt(hand_label)
        self.object_label_list = readTxt(object_label)
        self.dataset_size = len(self.img_list)
        self.transforms = transforms
        self.mesh = mesh
        self.bbox_3d = bbox_3d
        self.intrinsics = intrinsics
        self.extrinsic = extrinsics

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img_path = self.img_list[idx]

        object_mesh = None
        object_bbox_3d = None
        if "juice" in img_path:#0
            object_mesh = self.mesh['juice']
            object_bbox_3d = self.bbox_3d['juice']
        elif "milk" in img_path:#2
            object_mesh = self.mesh['milk']
            object_bbox_3d = self.bbox_3d['milk']
        elif 'salt' in img_path:#3
            object_mesh = self.mesh['salt']
            object_bbox_3d = self.bbox_3d['salt']
        elif 'soap' in img_path:#1
            object_mesh = self.mesh['liquid_soap']
            object_bbox_3d = self.bbox_3d['liquid_soap']

        # read original img, seg_laebl and reg_laebl
        data = Image.open(img_path)
        object_pose = str_to_np(self.object_label_list[idx])
        seg_label, binary_mask = prepare_seg_label(object_pose, object_mesh, self.intrinsics, self.extrinsic)  # np array
        reg_label = prepare_reg_label(object_pose, object_bbox_3d, self.intrinsics,self.extrinsic)  # (8,2), 0 for x(width), 1 for y(height)

        # crop using the seg_label
        data_crop, seg_label_crop, cord_upleft = image_label_crop(data, seg_label, binary_mask)  # PIL Image, np.array
        width, height = data_crop.size[0], data_crop.size[1]
        aligned_reg_label = align_reg_label(reg_label, cord_upleft)

        # resize to target size
        data_resize = data_crop.resize((config.resize_height, config.resize_width)) # to (608,608)
        seg_label_resize = cv2.resize(seg_label_crop, (config.resize_height,config.resize_width), interpolation=cv2.INTER_NEAREST)
        resize_reg_label = reshape_reg_label(aligned_reg_label, [width, height], [config.resize_width, config.resize_height])
        resize_reg_label = resize_reg_label / config.resize_height # normalize to [0,1]

        # resize seg laebl to (76,76)
        seg_label_train = cv2.resize(seg_label_resize, (config.grid_num,config.grid_num), interpolation=cv2.INTER_NEAREST)
        seg_label_train = seg_label_train.reshape((config.grid_num,config.grid_num,1))
        mask = np.ma.getmaskarray(np.ma.masked_not_equal(seg_label_train, 0)).astype(int)

        data_train = self.transforms(data_resize)
        seg_label_train = torch.squeeze(self.transforms(seg_label_train))

        sample = {'data': data_train, 'seg_label': seg_label_train, 'reg_label': resize_reg_label, 'mask': torch.squeeze(torch.from_numpy(mask).float())}
        return sample
