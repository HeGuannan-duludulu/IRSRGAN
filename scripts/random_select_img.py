
import os
import random

all_img_dir = r'../data/all_img'

all_pic_name = os.listdir(all_img_dir)
print(len(all_pic_name))
a = random.sample(all_pic_name, 15000)

output_dir = r'../data/CVC-09_14(train)/original'
remain_img_dir = r'../data/CVC-09_14(test)'

