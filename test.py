import os
import argparse
import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchsummary import summary

from ImagePairPrefixFolder import ImagePairPrefixFolder, var_custom_collate
from utils import make_dataset, edge_compute, ssim
from skimage import metrics

parser = argparse.ArgumentParser()
parser.add_argument('--network', default='AIENet')
parser.add_argument('--task', default='Fog', help='CompressionBlur | MotionBlur | Fog | all')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--indir', default='/home/baode/PycharmProjects/Datasets/archive/test_big_aug/test')
parser.add_argument('--gtdir', default='/home/baode/PycharmProjects/Datasets/archive/test_big_resize')
parser.add_argument('--outdir', default='results')
parser.add_argument('--log_name', default='log_test.txt')
parser.add_argument('--no_edge', action='store_true')
parser.add_argument('--only_residual', action='store_true', help='regress residual rather than image')
parser.add_argument('--test_max_size', type=int, default=1080)
parser.add_argument('--size_unit', type=int,  default=8)
parser.add_argument('--force_rgb', action='store_true')

opt = parser.parse_args()
assert opt.task in ['CompressionBlur', 'MotionBlur', 'Fog', 'all']

opt.model = ''
opt.use_cuda = opt.gpu_id >= 0
opt.outdir = os.path.join('checkpoint/%s/%s' % (opt.outdir, opt.task))

opt.log_name = os.path.join(opt.outdir, opt.log_name)
opt.outdir = os.path.join(opt.outdir)
if not os.path.exists(opt.outdir):
    os.makedirs(opt.outdir)
    if opt.task == 'all':
        os.mkdir(os.path.join(opt.outdir, 'CompressionBlur'))
        os.mkdir(os.path.join(opt.outdir, 'MotionBlur'))
        os.mkdir(os.path.join(opt.outdir, 'Fog'))
if opt.task != 'all':
    opt.indir = os.path.join(opt.indir, opt.task)

test_dataset = ImagePairPrefixFolder(opt.indir, opt.gtdir,
                                     max_img_size=opt.test_max_size, size_unit=opt.size_unit, force_rgb=opt.force_rgb)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             collate_fn=var_custom_collate, pin_memory=True,
                             num_workers=1)

if opt.network == 'AIENet':
    from AIENet import AIENet
    net = AIENet()
else:
    print('network structure %s not supported' % opt.network)
    raise ValueError

if opt.use_cuda:
    torch.cuda.set_device(opt.gpu_id)
    net.cuda()
else:
    net.float()

# net.load_state_dict(torch.load(opt.model)['net_state'])
net.load_state_dict(torch.load(opt.model))
net.eval()

avg_psnr = 0
avg_ssim = 0
task_cnt = 0
for iter, data in enumerate(test_dataloader):
    batch_input_img, batch_input_edge, batch_gt = data
    if opt.gpu_id > -1:
        batch_input_img, batch_input_edge, batch_gt = batch_input_img.cuda(), batch_input_edge.cuda(), batch_gt.cuda()

    if opt.no_edge:
        batch_input = batch_input_img
    else:
        batch_input = torch.cat((batch_input_img, batch_input_edge), dim=1)
    batch_input_v = Variable(batch_input)
    batch_gt_v = Variable(batch_gt)

    pred = net(batch_input_v)  # [N,C,H,W]

    if opt.only_residual:
        gt_arr = batch_gt_v.data[0].cpu().float().round().clamp(0, 255).numpy().astype(np.uint8).transpose(1, 2, 0)
        pred_arr = (pred + Variable(batch_input_img + 128)).data[0].cpu().float().round().clamp(0, 255).numpy().astype(np.uint8).transpose(1, 2, 0)
        loss_psnr = metrics.peak_signal_noise_ratio(gt_arr, pred_arr)
        loss_ssim = metrics.structural_similarity(gt_arr, pred_arr, multichannel=True)
    else:
        gt_arr = batch_gt_v.data[0].cpu().float().round().clamp(0, 255).numpy().astype(np.uint8).transpose(1, 2, 0)
        pred_arr = pred.data[0].cpu().float().round().clamp(0, 255).numpy().astype(np.uint8).transpose(1, 2, 0)
        loss_psnr = metrics.peak_signal_noise_ratio(gt_arr,pred_arr)
        loss_ssim = metrics.structural_similarity(gt_arr,pred_arr,multichannel=True)
    avg_psnr += loss_psnr
    avg_ssim += loss_ssim
    task_cnt += 1
    out_img = Image.fromarray(pred_arr)
    img_path = test_dataset.get_input_info(iter)
    if opt.task == 'all':
        out_img.save(os.path.join(opt.outdir, os.path.splitext(os.path.basename(img_path))[0] + '_%s.png' % opt.task))
    else:
        out_img.save(os.path.join(opt.outdir, img_path + '_%s.png' % opt.task))

    print('img %d:' % task_cnt +'  PSNR:%.6f  SSIM:%.6f ' % (loss_psnr, loss_ssim))

message_psnr = 'Average test PSNR is %f for %d images' % (avg_psnr / task_cnt, task_cnt)
message_ssim = 'Average test SSIM is %f for %d images' % (avg_ssim / task_cnt, task_cnt)
with open(opt.log_name, "a") as log_file:
    log_file.write('%s\n' % ('Testing for %s' % opt.model))
    log_file.write('%s\n' % message_psnr)
    log_file.write('%s\n' % message_ssim)
    log_file.write('\n')
