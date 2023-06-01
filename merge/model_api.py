"""
This file is an api to use super-resolution model

"""

import os

import cv2
import torch
from natsort import natsorted

import irsrgan_config
import imgproc
import model
from utils import make_directory

sr_dir = "./sr_dir/"
model_dir = "merge_model"
lr_img_dir = "./lr_dir/lr_img_dir"


def main(sr_image_path=None) -> None:
    # Initialize the super-resolution bsrgan_model
    irsrgan_model_ = model.Generator()
    irsrgan_model = irsrgan_model_.to(device=torch.device("cuda") if torch.cuda.is_available() else "cpu")
    print(f"Build `{irsrgan_config.g_arch_name}` model successfully.")

    # Load the super-resolution irsrgan_model weights
    checkpoint = torch.load(g_model_weights_path, map_location=lambda storage, loc: storage)
    irsrgan_model.load_state_dict(checkpoint["state_dict"])
    print(f"Load `{irsrgan_config.g_arch_name}` model weights "
          f"`{os.path.abspath(g_model_weights_path)}` successfully.")

    # Start the verification mode
    irsrgan_model.eval()

    # Get a list of test image file names.
    file_names = natsorted(os.listdir(lr_img_dir))
    # Get the number of test image files.
    total_files = len(file_names)

    for index in range(total_files):
        lr_image_path = os.path.join(lr_img_dir, file_names[index])
        # sr_image_path = os.path.join(sr_dir, file_names[index])
        print(file_names[index])
        print(f"Processing `{os.path.abspath(lr_image_path)}`...")
        lr_tensor = imgproc.preprocess_one_image(lr_image_path,
                                                 torch.device("cuda") if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            sr_tensor = irsrgan_model(lr_tensor)

        # Save image
        sr_image = imgproc.tensor_to_image(sr_tensor, False, False)
        sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(sr_image_path, file_names[index]), sr_image)


if __name__ == "__main__":
    model_list = os.listdir(model_dir)
    print(model_list)
    for model_name in model_list:
        g_model_weights_path = os.path.join(model_dir, model_name)
        path = os.path.join(sr_dir, model_name)
        make_directory(path)
        main(sr_image_path=path)
