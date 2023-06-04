"""Automatic image renaming"""

import os

path_name = r'./your_datasets'
# path_name: Represents the folder you need to change in bulk
i = 0
for item in os.listdir(path_name):  # Go inside the folder and iterate through each file
    os.rename(os.path.join(path_name, item), os.path.join(path_name, 'new_name_{}.png'.format(str(i).zfill(6))))
    i += 1
