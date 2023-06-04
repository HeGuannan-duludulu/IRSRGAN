"""
Calculate Lpips-I
"""

import lpips
import cv2
import numpy as np
import torch
import math


hr_img_path = "./test_dir/gt_dir/your_hr_img.jpg"
sr_img_path = "./test_dir/sr_dir/your_sr_img.jpg"

loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg')  # closer to "traditional" perceptual loss, when used for optimization

org = cv2.imread(hr_img_path).astype(np.float32) / 127.5 - 1
# image should be RGB, IMPORTANT: normalized to [-1,1]

input_img = cv2.imread(sr_img_path)

img1 = input_img.astype(np.float32) / 127.5 - 1
img0 = torch.from_numpy(org)
img1 = torch.from_numpy(img1)
img0 = img0.permute(2, 1, 0)
img1 = img1.permute(2, 1, 0)

# Calculate the blur degree of the image
imageVar = cv2.Laplacian(input_img, cv2.CV_64F).var()
# Calculate Lpips
d = loss_fn_vgg(img0, img1)

# calc Lpips-I
Lpips_I = math.log2(imageVar) / d.tolist()[0][0][0][0]
