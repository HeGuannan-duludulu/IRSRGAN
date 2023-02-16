import numpy as np
import random
import copy

from scipy import ndimage
import scipy
import cv2

# noinspection SpellCheckingInspection
from typing import Tuple




class Degradation:
    """
    input: 128X128X3
    output: 32X32X3
    """
    def __init__(self, scale_factor: int):
        self.sf = scale_factor
        self.downsamplerate = self.sf / 2
        self.size = 128

    def _add_gaussian_noise(self, img: np.ndarray) -> np.ndarray:
        mean = 0.1
        sigma = 0.1
        image = np.asarray(img / 255, dtype=np.float32)  # 图片灰度标准化
        noise = np.random.normal(mean, sigma, image.shape).astype(dtype=np.float32)  # 产生高斯噪声
        output = image + noise  # 将噪声和图片叠加
        output = np.clip(output, 0, 1)
        output = np.uint8(output * 255)
        return output

    def _add_JPEG_noise(self, img) -> np.ndarray:
        quality_factor = random.randint(70, 95)
        result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
        img = cv2.imdecode(encimg, 0)
        return img

    def add_blur(self, img):
        wd = 2.0 + 0.2 * self.sf
        k = self._fspecial('gaussian', 1 * random.randint(2, 5) + 3, wd * random.random())
        img = ndimage.filters.convolve(img, k, mode='mirror')
        return img

    @staticmethod
    def _fspecial_gaussian(hsize, sigma):
        hsize = [hsize, hsize]
        siz = [(hsize[0] - 1.0) / 2.0, (hsize[1] - 1.0) / 2.0]
        std = sigma
        [x, y] = np.meshgrid(np.arange(-siz[1], siz[1] + 1), np.arange(-siz[0], siz[0] + 1))
        arg = -(x * x + y * y) / (2 * std * std)
        h = np.exp(arg)
        h[h < scipy.finfo(float).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h = h / sumh
        return h

    @staticmethod
    def _fspecial_laplacian(alpha):
        print(alpha)
        alpha = max([0, min([alpha, 1])])
        h1 = alpha / (alpha + 1)
        h2 = (1 - alpha) / (alpha + 1)
        h = [[h1, h2, h1], [h2, -4 / (alpha + 1), h2], [h1, h2, h1]]
        h = np.array(h)
        return h

    def _fspecial(self, filter_type, *args, **kwargs):

        if filter_type == 'gaussian':
            return self._fspecial_gaussian(*args, **kwargs)
        if filter_type == 'laplacian':
            return self._fspecial_laplacian(*args, **kwargs)

    @staticmethod
    def _uint2single(img) -> np.ndarray:
        return np.float32(img / 255.)

    @staticmethod
    def _single2uint(img) -> np.ndarray:
        return np.uint8((img.clip(0, 1) * 255.).round())

    def _downsample(self, img) -> np.ndarray:
        if np.random.rand() < 1:
            img = cv2.resize(img, (int(1/self.downsamplerate * self.size), int(1/self.downsamplerate * self.size)),
                             interpolation=random.choice([1, 2, 3]))
        self.size = img.shape[0]
        return img

    def _single2three(self, img):
        img_src = np.expand_dims(img, axis=2)
        img_src = np.concatenate((img_src, img_src, img_src), axis=-1)
        return img_src

    def second_degradation(self, img: np.ndarray) -> np.ndarray:
        """
        This is the degradation model of BSRGAN from the paper
        "Designing a Practical Degradation Model for Deep Blind Image Super-Resolution"
        ----------
        img: HXWXC, [0, 1], its size should be large than (lq_patchsizexsf)x(lq_patchsizexsf)
        sf: scale factor


        Returns
        -------
        img: low-quality patch, size: lq_patchsizeXlq_patchsizeXC, range: [0, 1]
        hq: corresponding high-quality patch, size: (lq_patchsizexsf)X(lq_patchsizexsf)XC, range: [0, 1]
        """

        img = cv2.resize(img, (self.size, self.size))
        result_img = img

        degradation_dic = {
            '0': self.add_blur,
            '1': self.add_blur,
            '2': self._downsample,
            '3': self._downsample,
            '4': self._add_gaussian_noise,
            '5': self._add_JPEG_noise,
        }
        shuffle_order = random.sample(range(6), 6)
        for step_num in shuffle_order:
            # name = degradation_dic['{}'.format(step_num)]
            result_img = degradation_dic['{}'.format(step_num)](result_img)
        result_img = self._add_JPEG_noise(result_img)
        result_img = self._single2three(result_img)

        return result_img


if __name__ == '__main__':
    img_ = cv2.imread('../123.png', 0)
    img = Degradation(scale_factor=4).second_degradation(img_)
    cv2.imshow('1234', img)
    print(img.shape)
    cv2.waitKey()
