import sys
import torch
from collections import OrderedDict

alpha = 1.0

net_PSNR_path = './test_dir/irsrrdb-psnr/g_epoch_240.pth.tar'
net_ESRGAN_path = './test_dir/IRSRGAN_org_with_pretrained(random_deg)/g_epoch_145.pth.tar'
net_interp_path = './merge/interp_{:02d}.pth.tar'.format(int(alpha*10))

net_PSNR = torch.load(net_PSNR_path, map_location=torch.device('cpu'))
net_ESRGAN = torch.load(net_ESRGAN_path, map_location=torch.device('cpu'))
net_interp = OrderedDict()

print('Interpolating with alpha = ', alpha)
net_PSNR_ = net_PSNR["state_dict"]
net_ESRGAN_ = net_ESRGAN["state_dict"]
net_interp = {"state_dict": {}}

for k, v_PSNR in net_PSNR_.items():
    v_ESRGAN = net_ESRGAN_[k]
    #print(v_PSNR)
    net_interp["state_dict"][k] = (1 - alpha) * v_PSNR + alpha * v_ESRGAN

torch.save(net_interp, net_interp_path)
