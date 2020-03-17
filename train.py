import torch
from torch.autograd import Variable
import time
from segpose_net import SegPoseNet
from dataset import FPHA_hand
import config
from config import args_setting
from torchvision import transforms
from torch.optim import lr_scheduler
from utils import *

def train(args, epoch, model, train_loader, device, optimizer, seg_criterion, reg_criterion, root):
    since = time.time()
    model.train()

    for batch_idx,  sample_batched in enumerate(train_loader):
        data, mask_front, seg_label, reg_label = sample_batched['data'].to(device), \
                                                sample_batched['mask'].to(device),\
                                              sample_batched['seg_label'].type(torch.LongTensor).to(device), \
                                              sample_batched['reg_label'].type(torch.FloatTensor).to(device)# reg_label:(B, 8, 2), 2d bbox8 label
        ''''''
        optimizer.zero_grad()
        output = model(data)
        # foreground/background semantic seg result
        seg_confs = output[0][0] #(B,S,S), grid-level segmentaion confidence
        seg_ids = output[0][1] #(B,S,S), grid-level segmentaion class
        seg_out = output[0][2] #(B,2,S,S), grid-level segmentaion output, 2nd dimension is the number of class
        # 2d bbox8 regression result
        reg_px = output[1][0] #(B,S,S,8), grid-level x coordinate regression
        reg_py = output[1][1] #(B,S,S,8), grid-level y coordinate regression
        reg_conf = output[1][2] #(B,S,S,8), grid-level coordinate regression confidence

        # segmentation loss
        seg_loss = seg_criterion(seg_out, seg_label)

        # bbox-2d cord regression loss
        mask_front = mask_front.repeat(8, 1, 1, 1).permute(1, 2, 3, 0).contiguous()  # (B,S,S,8)
        reg_py = reg_py * mask_front  # (B,S,S,8)
        reg_px = reg_px * mask_front
        reg_label = reg_label.repeat(76, 76, 1, 1, 1).permute(2, 0, 1, 3, 4).contiguous() # (B,76,76,8,2)
        reg_label_x = reg_label[:, :, :, :, 0]  # # (B,S,S,8), normalize into [0,1]
        reg_label_y = reg_label[:, :, :, :, 1]
        reg_label_x = reg_label_x * mask_front
        reg_label_y = reg_label_y * mask_front
        reg_loss = reg_criterion(reg_px, reg_label_x) + reg_criterion(reg_py, reg_label_y)

        # regression confidence loss
        bias = torch.sqrt((reg_py - reg_label_y) ** 2 + (reg_px - reg_label_x) ** 2)
        conf_target = torch.exp(-config.modulatingFactor * bias) * mask_front
        conf_target = conf_target.detach()
        conf_loss = reg_criterion(reg_conf, conf_target)

        loss = config.alpha * seg_loss + config.gamma * reg_loss + config.beta * conf_loss
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSeg Loss: {:.6f}\tReg Loss: {:.6f}\tConf Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), seg_loss.item(), reg_loss.item(), conf_loss.item()))

        if batch_idx % 2000 == 0:
            save_name = os.path.join(root, 'model_%d_%d.pth' % (epoch, batch_idx))
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, save_name)

    time_elapsed = time.time() - since
    print('Train Epoch: {} complete in {:.0f}m {:.0f}s'.format(epoch,
        time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    args = args_setting()
    torch.manual_seed(args.seed)
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    save_root = './model'
    data_options = read_data_cfg(args.cfg_file)
    model = SegPoseNet(data_options).to(device)
    model.print_network()

    # turn image into floatTensor
    op_tranforms = transforms.Compose([transforms.ToTensor()])

    mesh = load_objects(config.object_model_root)
    bbox_3d = get_bbox8_3d_from_dict(mesh)

    # load data for batches, num_workers for multiprocess
    train_loader = torch.utils.data.DataLoader(
        FPHA_hand(file_path=config.train_img_path,
                  hand_label=config.train_hand_annotation,
                  object_label=config.train_object_annotation,
                  transforms=op_tranforms,
                  mesh=mesh,
                  bbox_3d=bbox_3d,
                  intrinsics= config.k_FPHA,
                  extrinsics=config.ex_FPHA),
        batch_size=args.batch_size, shuffle=True, num_workers=config.data_loader_numworkers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    class_weight = torch.Tensor(config.class_weight)
    seg_criterion = torch.nn.CrossEntropyLoss(weight=class_weight).to(device)
    reg_criterion = torch.nn.L1Loss().to(device) #reduction='mean'
    best_acc = 0

    # train
    for epoch in range(1, args.epochs+1):
        train(args, epoch, model, train_loader, device, optimizer, seg_criterion, reg_criterion, save_root)
        scheduler.step()
        #val(args, model, val_loader, device, criterion, best_acc)