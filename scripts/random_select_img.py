
import os
import random
import shutil

all_img_dir = r'../TISR_challenge_pic'
output_dir = r'../Essay/manuscript/EXP/EXP1/test_image'
not_selected_img_dir = r'..\data\CVC-09_14(test)'
select_num = 100

all_pic_name = os.listdir(all_img_dir)

random_selected = random.sample(all_pic_name, select_num)
not_selected = [img for img in all_pic_name if img not in random_selected]


for img_name in random_selected:
    shutil.copy(os.path.join(all_img_dir, img_name), os.path.join(output_dir, img_name))

for img_name in not_selected:
    shutil.copy(os.path.join(all_img_dir, img_name), os.path.join(not_selected_img_dir, img_name))

print(os.path.join(all_img_dir, random_selected[0]))
