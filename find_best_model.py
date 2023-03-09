import os

import cv2
import torch
from natsort import natsorted

import irsrgan_config
import imgproc
import model
from image_quality_evaluate import PSNR, SSIM
from utils import make_directory


model_dir = "./samples"
test_image_lr_dir = "./test_dir/TISR/lr"
test_image_hr_dir = "./test_dir/TISR/hr"
output_image_dir = "./test_dir/TISR/sr"

model_list = [model for model in os.listdir(model_dir) if model != "logs"]



def main() -> None:
    for model_name in model_list:
        print(f"Build `{model_name}` model successfully.")

        abs_path = os.path.join(model_dir, model_name)
        for each_model in os.listdir(abs_path):
            irsrgan_model_ = model.Generator()
            irsrgan_model = irsrgan_model_.to(device=irsrgan_config.device)
            each_model_path = os.path.join(abs_path, each_model)
            checkpoint = torch.load(each_model_path, map_location=lambda storage, loc: storage)
            irsrgan_model.load_state_dict(checkpoint["state_dict"])
            print(f"Load `{each_model}` model weights "f"`{each_model_path}` successfully.")

            epoch = each_model.strip("_")
            each_output_dir = os.path.join(output_image_dir, "{}_{}".format(model_name, epoch))
            make_directory(each_output_dir)

            irsrgan_model.eval()

            psnr = PSNR(irsrgan_config.upscale_factor, irsrgan_config.only_test_y_channel)
            ssim = SSIM(irsrgan_config.upscale_factor, irsrgan_config.only_test_y_channel)

            psnr = psnr.to(device=irsrgan_config.device, non_blocking=True)
            ssim = ssim.to(device=irsrgan_config.device, non_blocking=True)

            psnr_metrics = 0.0
            ssim_metrics = 0.0

            lr_list = os.listdir(test_image_lr_dir)
            hr_list = os.listdir(test_image_hr_dir)
            assert len(lr_list) == len(hr_list)

            for num in range(len(lr_list)):
                lr_image_path = os.path.join(test_image_lr_dir, lr_list[num])
                hr_image_path = os.path.join(test_image_hr_dir, hr_list[num])

                lr_tensor = imgproc.preprocess_one_image(lr_image_path, irsrgan_config.device)
                gt_tensor = imgproc.preprocess_one_image(hr_image_path, irsrgan_config.device)

                with torch.no_grad():
                    sr_tensor = irsrgan_model(lr_tensor)

                sr_image = imgproc.tensor_to_image(sr_tensor, False, False)
                sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(each_output_dir, lr_list[num]), sr_image)
                """
                psnr_metrics += psnr(sr_tensor, gt_tensor).item()
                ssim_metrics += ssim(sr_tensor, gt_tensor).item()

            avg_psnr = 100 if psnr_metrics / len(lr_list) > 100 else psnr_metrics / len(lr_list)
            avg_ssim = 1 if ssim_metrics / len(lr_list) > 1 else ssim_metrics / len(lr_list)

            print(f"PSNR: {avg_psnr:4.2f} [dB]\n"
                  f"SSIM: {avg_ssim:4.4f} [u]")
                  """
"""

    # Create a folder of super-resolution experiment results
    make_directory(irsrgan_config.sr_dir)

    # Start the verification mode of the bsrgan_model.
    irsrgan_model.eval()

    # Initialize the sharpness evaluation function
    psnr = PSNR(irsrgan_config.upscale_factor, irsrgan_config.only_test_y_channel)
    ssim = SSIM(irsrgan_config.upscale_factor, irsrgan_config.only_test_y_channel)

    # Set the sharpness evaluation function calculation device to the specified model
    psnr = psnr.to(device=irsrgan_config.device, non_blocking=True)
    ssim = ssim.to(device=irsrgan_config.device, non_blocking=True)

    # Initialize IQA metrics
    psnr_metrics = 0.0
    ssim_metrics = 0.0

    # Get a list of test image file names.
    file_names = natsorted(os.listdir(irsrgan_config.lr_dir))
    # Get the number of test image files.
    total_files = len(file_names)

    for index in range(total_files):
        lr_image_path = os.path.join(irsrgan_config.lr_dir, file_names[index])
        sr_image_path = os.path.join(irsrgan_config.sr_dir, file_names[index])
        gt_image_path = os.path.join(irsrgan_config.gt_dir, file_names[index])

        print(f"Processing `{os.path.abspath(lr_image_path)}`...")
        lr_tensor = imgproc.preprocess_one_image(lr_image_path, irsrgan_config.device)
        gt_tensor = imgproc.preprocess_one_image(gt_image_path, irsrgan_config.device)

        # Only reconstruct the Y channel image data.
        with torch.no_grad():
            sr_tensor = irsrgan_model(lr_tensor)

        # Save image
        sr_image = imgproc.tensor_to_image(sr_tensor, False, False)
        sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(sr_image_path, sr_image)

        # Cal IQA metrics
        psnr_metrics += psnr(sr_tensor, gt_tensor).item()
        ssim_metrics += ssim(sr_tensor, gt_tensor).item()


    avg_psnr = 100 if psnr_metrics / total_files > 100 else psnr_metrics / total_files
    avg_ssim = 1 if ssim_metrics / total_files > 1 else ssim_metrics / total_files

    print(f"PSNR: {avg_psnr:4.2f} [dB]\n"
          f"SSIM: {avg_ssim:4.4f} [u]")

"""


if __name__ == "__main__":
    main()

