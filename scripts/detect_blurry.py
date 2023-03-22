
import cv2
import numpy as np

def is_blurry(image_path):
    # 加载图像
    image = cv2.imread(image_path)
    # 计算拉普拉斯方差
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    gradient = np.sqrt(np.square(sobelx) + np.square(sobely))

    # 计算梯度能量
    energy = np.sum(gradient)
    # 如果方差小于阈值，则认为图像模糊
    threshold = 100
    return energy


if __name__ == "__main__":
    image_path = "../test_dir/blurry_infrared_image/1.png"
    v = is_blurry(image_path)
    print(v)
