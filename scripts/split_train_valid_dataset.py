import argparse
import os
import random
import shutil

from tqdm import tqdm


def main(args) -> None:
    if not os.path.exists(args.train_images_dir):
        os.makedirs(args.train_images_dir)
    if not os.path.exists(args.valid_images_dir):
        os.makedirs(args.valid_images_dir)

    train_files = os.listdir(args.train_images_dir)
    valid_files = random.sample(train_files, int(len(train_files) * args.valid_samples_ratio))

    process_bar = tqdm(valid_files, total=len(valid_files), unit="image", desc="Split")

    for image_file_name in process_bar:
        shutil.copyfile(f"{args.train_images_dir}/{image_file_name}", f"{args.valid_images_dir}/{image_file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split train and valid dataset scripts.")
    parser.add_argument("--train_images_dir", type=str, help="Path to train image directory.")
    parser.add_argument("--valid_images_dir", type=str, help="Path to valid image directory.")
    parser.add_argument("--valid_samples_ratio", type=float, help="What percentage of the data is extracted from the training set into the validation set.")
    args = parser.parse_args()

    main(args)