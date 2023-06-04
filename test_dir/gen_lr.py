"""
This code is used for generating LR img (128x128 -> 32x32)
"""

import cv2
import os

gt_dir = r'./gt_dir'
output_dir = './lr_dir'


for pic_name in os.listdir(gt_dir):
    each_pic = cv2.imread(os.path.join(gt_dir, pic_name))
    each_pic_ = cv2.resize(each_pic, (32, 32))
    cv2.imwrite(os.path.join(output_dir, pic_name), each_pic_)
