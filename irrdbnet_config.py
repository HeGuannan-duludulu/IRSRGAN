
import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)

np.random.seed(0)
# If GPU is not availble, then use cpu
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# When evaluating the performance of the SR model, whether to verify only the Y channel image data
only_test_y_channel = True
# Model architecture name
g_arch_name = "rrdbnet_x4"
# Model arch config
in_channels = 3
out_channels = 3
channels = 64
growth_channels = 32
num_blocks = 23
upscale_factor = 4
# Current configuration parameter method
mode = "train"
# Experiment name, easy to save weights and log files
exp_name = "IRRDBNet_x4-psnr"

if mode == "train":
    # Dataset address
    train_gt_images_dir = f"./data/IRSRGAN/train"

    test_gt_images_dir = f"./data/IRSRGAN/valid"
    #test_lr_images_dir = f"./data/Set5/LRbicx{upscale_factor}"

    gt_image_size = 128
    batch_size = 32
    num_workers = 4

    # The address to load the pretrained model
    pretrained_g_model_weights_path = ""

    # Incremental training and migration training
    resume_g_model_weights_path = f"./samples/IRRDBNet_x4-psnr/g_epoch_100.pth.tar"

    # Total num epochs (200 iters)
    epochs = 150

    # loss function weights
    loss_weights = 1.0

    # Optimizer parameter
    model_lr = 2e-4
    model_betas = (0.9, 0.99)
    model_eps = 1e-8
    model_weight_decay = 0.0

    # EMA parameter
    model_ema_decay = 0.99998

    # Dynamically adjust the learning rate policy
    lr_scheduler_step_size = epochs // 5
    lr_scheduler_gamma = 0.5

    # How many iterations to print the training result
    train_print_frequency = 10
    valid_print_frequency = 10

if mode == "test":
    # Test data address
    lr_dir = f"./test_dir/lr_dir"
    sr_dir = f"./results/test/{exp_name}"
    gt_dir = "./test_dir/gt_dir"

    g_model_weights_path = "./results/pretrained_models/RRDBNet_x4-DFO2K-2e2a91f4.pth.tar"
