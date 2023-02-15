


import torch
from torch import nn

generator = torch.load('../test_dir/full_IRSRGAN/g_epoch_10.pth.tar',map_location=lambda storage, loc: storage)
discriminator = torch.load('../test_dir/full_IRSRGAN/d_epoch_10.pth.tar',map_location=lambda storage, loc: storage)


model_dict = {'generator': generator, 'discriminator': discriminator}
model = nn.ModuleDict(model_dict)

#torch.save(model.state_dict(), 'gan.pth.tar')
