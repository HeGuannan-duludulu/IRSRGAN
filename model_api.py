"""
Super-resolution model api
"""

import os

import cv2
import torch
from natsort import natsorted

import irsrgan_config
import imgproc
import model
from utils import make_directory

sr_dir = "./test_dir/your_sr_dir"


def main() -> None:
    # Initialize the super-resolution bsrgan_model
    irsrgan_model_ = model.Generator()
    irsrgan_model = irsrgan_model_.to(device=irsrgan_config.device)
    print(f"Build `{irsrgan_config.g_arch_name}` model successfully.")

    # Load the super-resolution irsrgan_model weights
    checkpoint = torch.load(g_model_weights_path, map_location=lambda storage, loc: storage)
    irsrgan_model.load_state_dict(checkpoint["state_dict"])
    print(f"Load `{irsrgan_config.g_arch_name}` model weights "
          f"`{os.path.abspath(g_model_weights_path)}` successfully.")

    # Create a folder of super-resolution experiment results
    make_directory(sr_dir)

    # Start the verification mode
    irsrgan_model.eval()

    # Get a list of test image file names.
    file_names = natsorted(os.listdir(irsrgan_config.lr_dir))
    # Get the number of test image files.
    total_files = len(file_names)

    for index in range(total_files):
        lr_image_path = os.path.join(irsrgan_config.lr_dir, file_names[index])
        sr_image_path = os.path.join(sr_dir, file_names[index])
        print(file_names[index])

        print(f"Processing `{os.path.abspath(lr_image_path)}`...")
        lr_tensor = imgproc.preprocess_one_image(lr_image_path, irsrgan_config.device)

        with torch.no_grad():
            sr_tensor = irsrgan_model(lr_tensor)

        # Save super-resolution image
        sr_image = imgproc.tensor_to_image(sr_tensor, False, False)
        sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(sr_image_path, sr_image)


if __name__ == "__main__":
    if not os.path.exists(sr_dir):
        make_directory(sr_dir)
    model_list = os.listdir(sr_dir)
    g_model_weights_path = "./test_dir/models/IRSRGAN_org_with_pretrained/g_epoch_140.pth.tar"
    main()
