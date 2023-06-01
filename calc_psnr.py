from math import log10, sqrt
import cv2
import numpy as np
import os


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def main():
    original = cv2.imread("./test_dir/gt_dir/0601.jpg")
    for each_pic_name in os.listdir("./temp"):
        full_path = os.path.join("./temp", each_pic_name)
        each_pic = cv2.imread(full_path)
        value = PSNR(original, each_pic)
        print("PSNR value of {} is {} dB".format(each_pic_name, value))


if __name__ == "__main__":
    main()
