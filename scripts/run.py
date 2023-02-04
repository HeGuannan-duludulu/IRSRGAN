
import os

# Prepare dataset
os.system("python3 ./prepare_dataset.py --images_dir ../data/DIV2K/CVC-09_14(train)/DIV2K_train_HR --output_dir "
          "../data/DIV2K/ESRGAN/train --image_size 544 --step 272 --num_workers 16")
os.system("python3 ./prepare_dataset.py --images_dir ../data/DIV2K/CVC-09_14(train)/DIV2K_valid_HR --output_dir "
          "../data/DIV2K/ESRGAN/valid --image_size 544 --step 544 --num_workers 16")
