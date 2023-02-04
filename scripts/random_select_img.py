
import os
import random
import shutil

all_img_dir = r'..\data\all_img'
output_dir = r'..\data\CVC-09_14(train)\original'
not_selected_img_dir = r'..\data\CVC-09_14(test)'

all_pic_name = os.listdir(all_img_dir)

random_selected = random.sample(all_pic_name, 15000)
not_selected = [img for img in all_pic_name if img not in random_selected]


for img_name in random_selected:
    shutil.copy(os.path.join(all_img_dir, img_name), os.path.join(output_dir, img_name))

for img_name in not_selected:
    shutil.copy(os.path.join(all_img_dir, img_name), os.path.join(not_selected_img_dir, img_name))

print(os.path.join(all_img_dir, random_selected[0]))
