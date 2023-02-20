
import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# When evaluating the performance of the SR model, whether to verify only the Y channel image data
only_test_y_channel = True
# Model architecture name
d_arch_name = "discriminator"
g_arch_name = "irrdbnet_x4"
# Model arch config
in_channels = 3
out_channels = 3
channels = 64
growth_channels = 32
num_blocks = 23
upscale_factor = 4
# Current configuration parameter method
mode = "test"
# Experiment name, easy to save weights and log files
exp_name = "IRSRGAN_x4"

if mode == "train":
    # Dataset address
    train_gt_images_dir = f"./data/IRSRGAN/train"

    test_gt_images_dir = f"./data/IRSRGAN/valid"
    #test_lr_images_dir = f"./data/Set5/LRbicx{upscale_factor}"

    gt_image_size = 128
    batch_size = 32
    num_workers = 8

    # The address to load the pretrained model
    pretrained_d_model_weights_path = ""
    pretrained_g_model_weights_path = ""

    # Incremental training and migration training
    resume_d_model_weights_path = f""
    resume_g_model_weights_path = f""

    # Total num epochs (400,000 iters)
    epochs = 100

    # Loss function weight
    pixel_weight = 0.01
    content_weight = 1.0
    adversarial_weight = 0.005

    # Feature extraction layer parameter configuration
    feature_model_extractor_node = "features.34"
    feature_model_normalize_mean = [0.485, 0.456, 0.406]
    feature_model_normalize_std = [0.229, 0.224, 0.225]

    # Optimizer parameter
    model_lr = 2e-4
    model_betas = (0.9, 0.99)
    model_eps = 1e-8
    model_weight_decay = 0.0

    # EMA parameter
    model_ema_decay = 0.99998

    # Dynamically adjust the learning rate policy
    lr_scheduler_milestones = [int(epochs * 0.125), int(epochs * 0.250), int(epochs * 0.500), int(epochs * 0.750)]
    lr_scheduler_gamma = 0.5

    # How many iterations to print the training result
    train_print_frequency = 10
    valid_print_frequency = 10

if mode == "test":
    # Test data address
    lr_dir = f"./test_dir/lr_dir"
    sr_dir = f"./test_dir/sr_dir/{exp_name}"
    gt_dir = "./test_dir/gt_dir"

    g_model_weights_path = "./test_dir/irsrrdb-psnr/g_best.pth.tar"
