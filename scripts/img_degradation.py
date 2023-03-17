import numpy as np
import random
import copy

import cv2

__all__ = ["random_degradation"]

"""
input: 128X128X3
output: 32X32X3
"""
#num = 1
#blur_num = 1
#jpeg_num = 1

def _add_gaussian_noise(img: np.ndarray) -> np.ndarray:
    """
    Add gaussian noise to the image
    :param img: input image size(128x128)
    :return: image with gaussian noise
    """
    mean = 0.01
    sigma = 0.01
    image = np.asarray(img / 255, dtype=np.float32)  # 图片灰度标准化
    noise = np.random.normal(mean, sigma, image.shape).astype(dtype=np.float32)  # 产生高斯噪声
    output = image + noise  # 将噪声和图片叠加
    output = np.clip(output, 0, 1)
    output = np.uint8(output * 255)
    #cv2.imwrite('gas.png', output)
    return output


def _add_JPEG_noise(img) -> np.ndarray:
    """
    Add jpeg compression noise to the image, the image quality will be selected from 70 - 95 randomly
    :param img: input image
    :return: image with jpeg noise
    """
    #global jpeg_num
    quality_factor = random.randint(70, 95)
    result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.imdecode(encimg, cv2.IMREAD_UNCHANGED)
    #cv2.imwrite('jpeg{}.png'.format(jpeg_num), img)
    #jpeg_num += 1
    return img


def _add_blur(img):
    """
    Add blur to the image
    :param img: input image
    :return: blurry image
    """
    #global blur_num
    gausBlur = cv2.GaussianBlur(img, (3, 3), 0)
    #cv2.imwrite('blur{}.png'.format(blur_num), gausBlur)
    #blur_num += 1
    return gausBlur


def _uint2single(img) -> np.ndarray:
    """
    Made each pixel image from [0, 1]
    :param img: input image
    :return: [0, 1] image
    """
    return img.astype(np.float32) / 255


def _single2uint(img) -> np.ndarray:
    return img.astype(np.float32) * 255.


def _downsample(img) -> np.ndarray:
    """
    Down scale the image with 0.5 scale factor
    :param img: input image
    :return: 0.5 * image_org_size, 0.5 * image_org_size
    """
    #global num
    if True:
        img = cv2.resize(img, None, fx=0.5, fy=0.5,
                         interpolation=random.choice([1, 2, 3]))
    #cv2.imwrite('downsample{}.png'.format(num), img)
    #num += 1
    return img


def _single2three(img):
    """
    Convert single channel image into CxHxW format.
    :param img: input gray scale image
    :return: three channel image
    """
    img_src = np.expand_dims(img, axis=2)
    img_src = np.concatenate((img_src, img_src, img_src), axis=-1)
    return img_src


def random_degradation(image: np.ndarray) -> np.ndarray:
    """
    random degradation
    :param image: image.astype(np.float32)/255  size(Cx128x128)
    :return: size(Cx32x32)
    """
    result_img = image
    result_img = _single2uint(result_img)
    degradation_dic = {
        '0': _add_blur,
        '1': _add_blur,
        '2': _downsample,
        '3': _downsample,
        '4': _add_gaussian_noise,
        '5': _add_JPEG_noise,
    }
    shuffle_order = random.sample(range(6), 6)
    #print(shuffle_order)
    for step_num in shuffle_order:
        result_img = degradation_dic['{}'.format(step_num)](result_img)
    result_img = _add_JPEG_noise(result_img)
    #result_img = _uint2single(result_img)

    return result_img


if __name__ == '__main__':
    img_ = cv2.imread('../test_dir/gt_dir/1234.png').astype(np.float32) / 255
    print(img_.shape)
    img_2 = random_degradation(img_)
    print(img_2.shape)
    cv2.imwrite('final.png', img_2.astype(np.float32) * 255)
    cv2.waitKey()
