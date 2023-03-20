
import argparse
import os
import cv2
import torch
from natsort import natsorted

import irsrgan_config
import imgproc
import model
from utils import make_directory


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

    # Start the verification mode of the bsrgan_model.
    irsrgan_model.eval()



    lr_tensor = imgproc.preprocess_one_image(lr_image_path, irsrgan_config.device)


    # Only reconstruct the Y channel image data.
    with torch.no_grad():
        sr_tensor = irsrgan_model(lr_tensor)

    # Save image
    sr_image = imgproc.tensor_to_image(sr_tensor, False, False)
    sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(sr_image_path, sr_image)











video_path = ""

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()




if __name__ == "__main__":
    pass
