
import cv2
import os

gt_dir = r'./gt_dir'
output_dir = './lr_dir'
for pic in os.listdir(gt_dir):
    print(pic)
    each_pic = cv2.imread(os.path.join(gt_dir, pic))
    each_pic_ = cv2.resize(each_pic, (32, 32))
    cv2.imwrite(os.path.join(output_dir, pic), each_pic_)
