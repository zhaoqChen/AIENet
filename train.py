import os
import datetime
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from ImagePairPrefixFolder import ImagePairPrefixFolder, var_custom_collate
from utils import MovingAvg
from tf_visualizer import TFVisualizer
from PIL import Image
from skimage import metrics
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

#---------network setting---------#
parser.add_argument('--network', default='AIENet')
parser.add_argument('--name', default='all',help='CompressionBlur|Fog|MotionBlur|all')
parser.add_argument('--gpu_ids', default='0')
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_step', type=int, default=40)
parser.add_argument('--lr_gamma', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=0.0005)

#---------save dir---------------#
parser.add_argument('--checkpoints_dir', default='./checkpoint')
parser.add_argument('--logDir', default='tblogdir')

#---------save checkpoints setting-------------#
parser.add_argument('--resume_dir', default='')
parser.add_argument('--resume_epoch', type=int, default=0)
parser.add_argument('--save_epoch', type=int, default=5)
parser.add_argument('--save_latest_freq', type=int, default=5000)
parser.add_argument('--test_epoch', type=int, default=5)
parser.add_argument('--test_max_size', type=int, default=1080)
parser.add_argument('--size_unit', type=int,  default=8)
parser.add_argument('--print_iter', type=int,  default=100)

#---------dataset folder---------#
parser.add_argument('--input_folder', default='/home/baode/PycharmProjects/Datasets/archive/test_big_aug/train') 
parser.add_argument('--gt_folder', default='/home/baode/PycharmProjects/Datasets/archive/test_big_resize') 
parser.add_argument('--test_input_folder', default='/home/baode/PycharmProjects/Datasets/archive/test_big_aug/test')
parser.add_argument('--test_gt_folder', default='/home/baode/PycharmProjects/Datasets/archive/test_big_resize')

#---------para setting---------------#
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--only_residual', action='store_true', help='regress residual rather than image')
parser.add_argument('--loss_func', default='l2', help='l2|l1')
parser.add_argument('--inc', type=int, default=3)
parser.add_argument('--outc', type=int, default=3)
parser.add_argument('--force_rgb', action='store_true')
parser.add_argument('--no_edge', default = True, action='store_true')

opt = parser.parse_args()

# train folder
opt.input_folder = os.path.expanduser(opt.input_folder)
opt.gt_folder = os.path.expanduser(opt.gt_folder)

# test folder
opt.test_input_folder = os.path.expanduser(opt.test_input_folder)
opt.test_gt_folder = os.path.expanduser(opt.test_gt_folder)

if opt.name != 'all':
    opt.input_folder = os.path.join(opt.input_folder, opt.name)
    opt.test_input_folder = os.path.join(opt.test_input_folder, opt.name)


if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.name)):
    os.makedirs(os.path.join(opt.checkpoints_dir, opt.name))
opt.resume_dir = opt.resume_dir if opt.resume_dir != '' else os.path.join(opt.checkpoints_dir, opt.name)

visualizer = TFVisualizer(opt)

## print opt argument
for key, val in vars(opt).items():
    visualizer.print_logs('%s: %s' % (key, val))

opt.gpu_ids = [int(x) for x in opt.gpu_ids.split(',')]
torch.cuda.set_device(opt.gpu_ids[0])

# ----------------train dataset loader -------------------- #
train_dataset = ImagePairPrefixFolder(opt.input_folder, opt.gt_folder,size_unit=opt.size_unit, force_rgb=opt.force_rgb)
train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                              collate_fn=var_custom_collate, pin_memory=True,
                              num_workers=opt.num_workers)

# ----------------test dataset loader -------------------- #
opt.do_test = opt.test_gt_folder != ''
if opt.do_test:
    test_dataset = ImagePairPrefixFolder(opt.test_input_folder, opt.test_gt_folder,
                                         max_img_size=opt.test_max_size, size_unit=opt.size_unit, force_rgb=opt.force_rgb)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                 collate_fn=var_custom_collate, pin_memory=True,
                                 num_workers=1)

total_inc = opt.inc if opt.no_edge else opt.inc + 1

# ----------------network initialize -------------------- #
anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

if opt.network == 'AIENet':
    from AIENet import AIENet
    net = dehaze5(anchors_mask, 20, 'l', pretrained=False)
else:
    print('network structure %s not supported' % opt.network)
    raise ValueError
    
# ---------------- loss  -------------------- #
if opt.loss_func == 'l2':
    loss_crit = torch.nn.MSELoss()
elif opt.loss_func == 'l1':
    loss_crit = torch.nn.SmoothL1Loss()
else:
    print('loss_func %s not supported' % opt.loss_func)
    raise ValueError
pnsr_crit = torch.nn.MSELoss()

if len(opt.gpu_ids) > 0:
    net.cuda()
    if len(opt.gpu_ids) > 1:
        net = torch.nn.DataParallel(net)
    loss_crit = loss_crit.cuda()
    pnsr_crit = pnsr_crit.cuda()

optimizer = optim.Adam(net.parameters(), lr=opt.lr)
step_optim_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=opt.lr_gamma)
loss_avg = MovingAvg(pool_size=50)

start_epoch = 0
total_iter = 0

if opt.resume_epoch > 0:
    start_epoch = opt.resume_epoch
    total_iter = opt.resume_epoch * len(train_dataloader)
    resume_path = os.path.join(opt.resume_dir, 'net_epoch_%d.pth') % opt.resume_epoch
    print('resume from : %s' % resume_path)
    assert os.path.exists(resume_path), 'cannot find the resume model: %s ' % resume_path
    if isinstance(net, torch.nn.DataParallel):
        net.module.load_state_dict(torch.load(resume_path))
    else:
        net.load_state_dict(torch.load(resume_path))

# ---------------- start training -------------------- #

if __name__ == '__main__':
    train_psnr = []
    train_ssim = []
    for epoch in range(start_epoch, opt.epochs):
        visualizer.print_logs("Start to train epoch %d" % epoch)
        net.train()
        for iter, data in enumerate(train_dataloader):
            total_iter += 1
            optimizer.zero_grad()
            step_optim_scheduler.step(epoch)

            batch_input_img, batch_input_edge,  batch_gt = data
            if len(opt.gpu_ids) > 0:
                batch_input_img, batch_input_edge, batch_gt = batch_input_img.cuda(), batch_input_edge.cuda(), batch_gt.cuda()

            if opt.no_edge:
                batch_input = batch_input_img
            else:
                batch_input = torch.cat((batch_input_img, batch_input_edge), dim=1)
            batch_input_v = Variable(batch_input)
            if opt.only_residual:
                batch_gt_v = Variable(batch_gt - (batch_input_img+128))
            else:
                batch_gt_v = Variable(batch_gt)

            J_x = net(batch_input_v)

            loss = loss_crit(J_x, batch_gt_v)
            avg_loss = loss_avg.set_curr_val(loss.data)

            loss.backward()

            optimizer.step()

            if iter % opt.print_iter == 0:
                visualizer.plot_current_losses(total_iter, { 'loss': loss})
                visualizer.print_logs('epoch[%d]: Step[%d/%d], lr: %f, mv_avg_loss: %f, loss: %f' %
                                        (epoch, iter, len(train_dataloader),
                                         step_optim_scheduler.get_lr()[0], avg_loss, loss))
            if total_iter % opt.save_latest_freq == 0:
                latest_info = {'total_iter': total_iter,
                               'epoch': epoch,
                               'optim_state': optimizer.state_dict()}
                if len(opt.gpu_ids) > 1:
                    latest_info['net_state'] = net.module.state_dict()
                else:
                    latest_info['net_state'] = net.state_dict()
                print('save lastest model.')
                torch.save(latest_info, os.path.join(opt.checkpoints_dir, opt.name, 'latest.pth'))

        if (epoch+1) % opt.save_epoch == 0 :
            visualizer.print_logs('saving model for epoch %d' % epoch)
            if len(opt.gpu_ids) > 1:
                torch.save(net.module.state_dict(), os.path.join(opt.checkpoints_dir, opt.name, 'net_epoch_%d.pth' % (epoch+1)))
            else:
                torch.save(net.state_dict(), os.path.join(opt.checkpoints_dir, opt.name, 'net_epoch_%d.pth' % (epoch + 1)))

        if opt.do_test:
            avg_psnr = 0
            avg_ssim = 0
            task_cnt = 0
            net.eval()
            with torch.no_grad():
                for iter, data in enumerate(test_dataloader):
                    batch_input_img, batch_input_edge,  batch_gt = data
                    if len(opt.gpu_ids) > 0:
                        batch_input_img, batch_input_edge, batch_gt = batch_input_img.cuda(), batch_input_edge.cuda(), batch_gt.cuda()

                    if opt.no_edge:
                        batch_input = batch_input_img
                    else:
                        batch_input = torch.cat((batch_input_img, batch_input_edge), dim=1)
                    batch_input_v = Variable(batch_input)
                    batch_gt_v = Variable(batch_gt)


                    pred = net(batch_input_v)   #[N,C,H,W]

                    if opt.only_residual:
                        gt_arr = batch_gt_v.data[0].cpu().float().round().clamp(0, 255).numpy().astype(np.uint8).transpose(1, 2, 0)
                        pred_arr = (pred+Variable(batch_input_img+128)).data[0].cpu().float().round().clamp(0, 255).numpy().astype(np.uint8).transpose(1, 2, 0)
                        loss_psnr = metrics.peak_signal_noise_ratio(gt_arr,pred_arr)
                        loss_ssim = metrics.structural_similarity(gt_arr,pred_arr,multichannel=True)
                    else:
                        gt_arr = batch_gt_v.data[0].cpu().float().round().clamp(0, 255).numpy().astype(np.uint8).transpose(1, 2, 0)
                        pred_arr = pred.data[0].cpu().float().round().clamp(0, 255).numpy().astype(np.uint8).transpose(1, 2, 0)
                        loss_psnr = metrics.peak_signal_noise_ratio(gt_arr,pred_arr)
                        loss_ssim = metrics.structural_similarity(gt_arr,pred_arr,multichannel=True)
                    avg_psnr += loss_psnr
                    avg_ssim += loss_ssim
                    task_cnt += 1
            train_psnr.append(avg_psnr/task_cnt)
            train_ssim.append(avg_ssim/task_cnt)
            log_path = os.path.join(opt.checkpoints_dir, opt.name+'/psnr_ssim_log.txt')
            with open(log_path,'a') as train_log:
              train_log.write('Epoch[%d]  PSNR: %.6f   SSIM: %.6f\n' % (epoch, avg_psnr/task_cnt, avg_ssim/task_cnt))
            
            visualizer.print_logs('Testing for epoch: %d' % epoch)
            visualizer.print_logs('Average test PNSR is %f for %d images' % (avg_psnr/task_cnt, task_cnt))
            
    train_psnr = np.asfarray(train_psnr,float)
    train_ssim = np.asfarray(train_ssim,float)
    
    # plot 1:PSNR
    plt.figure()
    plt.plot(train_psnr,color='red',linewidth=1, linestyle='solid', label="PSNR")
    plt.tick_params(direction='in')
    plt.legend()
    plt.xlim([0,len(train_psnr)])
    plt.ylim([0,40])
    plt.axis('tight')
    plt.xlabel('Epoch',size=10)
    plt.ylabel('PSNR',size=10)
    plt.title(opt.name)
    plt.savefig(os.path.join(opt.checkpoints_dir, opt.name +'/PSNR.jpg'), dpi=800, bbox_inches='tight')
    
    # plot 2:SSIM
    plt.figure()
    plt.plot(train_ssim,linewidth=1, linestyle='solid', label="SSIM")
    plt.legend()
    plt.xlim([0,len(train_ssim)])
    plt.ylim([0,1])
    plt.axis('tight')
    plt.xlabel('Epoch',size=10)
    plt.ylabel('SSIM',size=10)
    plt.title(opt.name)
    plt.savefig(os.path.join(opt.checkpoints_dir, opt.name +'/SSIM.jpg'), dpi=800, bbox_inches='tight')
