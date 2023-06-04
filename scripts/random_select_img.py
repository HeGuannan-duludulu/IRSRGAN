"""
Random select "select_number" images of image datasets, and copy the selected images &  not selected images to
corresponding dir.
"""

import os
import random
import shutil

all_img_dir = r'../your_whole_datasets'
selected_dir = r'../selected_img_dir'
not_selected_img_dir = r'../not_selected_dir'

# select 100 random images
select_number = 100


all_pic_name = os.listdir(all_img_dir)

random_selected = random.sample(all_pic_name, select_number)
not_selected = [img for img in all_pic_name if img not in random_selected]


for img_name in random_selected:
    shutil.copy(os.path.join(all_img_dir, img_name), os.path.join(selected_dir, img_name))

for img_name in not_selected:
    shutil.copy(os.path.join(all_img_dir, img_name), os.path.join(not_selected_img_dir, img_name))

print(os.path.join(all_img_dir, random_selected[0]))
