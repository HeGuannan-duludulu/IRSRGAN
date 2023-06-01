import torch

Lambda = 1.0

net_PSNR_path = './test_dir/models/irsrrdb-psnr/g_epoch_240.pth.tar'
net_ESRGAN_path = './test_dir/models/IRSRGAN_org_with_pretrained(random_deg)/g_epoch_145.pth.tar'
net_interp_path = './merge/interp_{:02d}.pth.tar'.format(int(Lambda*10))

net_PSNR = torch.load(net_PSNR_path, map_location=torch.device('cpu'))
net_ESRGAN = torch.load(net_ESRGAN_path, map_location=torch.device('cpu'))


print('Interpolating with alpha = ', Lambda)
net_PSNR_ = net_PSNR["state_dict"]
net_ESRGAN_ = net_ESRGAN["state_dict"]
net_interp = {"state_dict": {}}

for k, v_PSNR in net_PSNR_.items():
    v_ESRGAN = net_ESRGAN_[k]
    net_interp["state_dict"][k] = (1 - Lambda) * v_PSNR + Lambda * v_ESRGAN

torch.save(net_interp, net_interp_path)
