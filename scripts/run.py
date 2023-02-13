
import os

# Prepare dataset
os.system("python3 ./prepare_dataset.py --images_dir ../data/all_img --output_dir "
          "../data/IRSRGAN/train --image_size 128 --num_workers 16")
os.system("python3 ./split_train_valid_dataset.py --train_images_dir ../data/IRSRGAN/train --valid_images_dir "
          "../data/IRSRGAN/valid --valid_samples_ratio 0.1")
