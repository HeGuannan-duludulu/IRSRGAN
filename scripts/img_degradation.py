import numpy as np
import random
from scipy import ndimage
import scipy
import cv2


# noinspection SpellCheckingInspection
class Degradation:
    def __init__(self, img: np.ndarray, scale_factor: int):
        self.img = img
        self.sf = scale_factor
        pass

    def add_gaussian_noise(self):
        mean = 0.1
        sigma = 0.1
        image = np.asarray(self.img / 255, dtype=np.float32)  # 图片灰度标准化
        noise = np.random.normal(mean, sigma, image.shape).astype(dtype=np.float32)  # 产生高斯噪声
        output = image + noise  # 将噪声和图片叠加
        output = np.clip(output, 0, 1)
        output = np.uint8(output * 255)
        return output

    def add_JPEG_noise(self):
        print('input_img:', self.img)
        quality_factor = random.randint(70, 95)
        # img = single2uint(img)
        result, encimg = cv2.imencode('.jpg', self.img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
        img = cv2.imdecode(encimg, 0)
        print('after_jpeg', img)
        # img = cv2.cvtColor(uint2single(img), cv2.COLOR_BGR2RGB)
        return img

    def add_blur(self, sf=4):
        wd = 2.0 + 0.2 * sf
        k = self.fspecial('gaussian', 1 * random.randint(2, 5) + 3, wd * random.random())

        img = ndimage.filters.convolve(self.img, k, mode='mirror')

        return img

    def fspecial_gaussian(self, hsize, sigma):
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

    def fspecial_laplacian(self, alpha):
        print(alpha)
        alpha = max([0, min([alpha, 1])])
        h1 = alpha / (alpha + 1)
        h2 = (1 - alpha) / (alpha + 1)
        h = [[h1, h2, h1], [h2, -4 / (alpha + 1), h2], [h1, h2, h1]]
        h = np.array(h)
        return h

    def fspecial(self, filter_type, *args, **kwargs):
        '''
        python code from:
        https://github.com/ronaldosena/imagens-medicas-2/blob/40171a6c259edec7827a6693a93955de2bd39e76/Aulas/aula_2_-_uniform_filter/matlab_fspecial.py
        '''
        if filter_type == 'gaussian':
            return self.fspecial_gaussian(*args, **kwargs)
        if filter_type == 'laplacian':
            return self.fspecial_laplacian(*args, **kwargs)

    def uint2single(self):
        return np.float32(self.img / 255.)

    def single2uint(self):
        return np.uint8((self.img.clip(0, 1) * 255.).round())

    def random_crop(self, lq, hq, sf=4, lq_patchsize=64):
        h, w = lq.shape[:2]
        rnd_h = random.randint(0, h - lq_patchsize)
        rnd_w = random.randint(0, w - lq_patchsize)
        lq = lq[rnd_h:rnd_h + lq_patchsize, rnd_w:rnd_w + lq_patchsize]

        rnd_h_H, rnd_w_H = int(rnd_h * sf), int(rnd_w * sf)
        hq = hq[rnd_h_H:rnd_h_H + lq_patchsize * sf, rnd_w_H:rnd_w_H + lq_patchsize * sf]
        return lq, hq

    def degradation_bsrgan(self, sf=4, lq_patchsize=72, isp_model=None):
        """
        This is the degradation model of BSRGAN from the paper
        "Designing a Practical Degradation Model for Deep Blind Image Super-Resolution"
        ----------
        img: HXWXC, [0, 1], its size should be large than (lq_patchsizexsf)x(lq_patchsizexsf)
        sf: scale factor
        isp_model: camera ISP model

        Returns
        -------
        img: low-quality patch, size: lq_patchsizeXlq_patchsizeXC, range: [0, 1]
        hq: corresponding high-quality patch, size: (lq_patchsizexsf)X(lq_patchsizexsf)XC, range: [0, 1]
        """
        isp_prob, jpeg_prob, scale2_prob = 0.25, 0.9, 0.25
        sf_ori = sf

        h1, w1 = self.img.shape[:2]
        # print(h1, w1)
        img = self.img.copy()[:h1 - h1 % sf, :w1 - w1 % sf, ...]  # mod crop
        h, w = img.shape[:2]
        # print(h, w)

        if h < lq_patchsize * sf or w < lq_patchsize * sf:
            raise ValueError(f'img size ({h1}X{w1}) is too small!')

        hq = img.copy()

        if sf == 4 and random.random() < scale2_prob:  # downsample1
            # if np.random.rand() < 0.5:
            if np.random.rand() < 1:
                img = cv2.resize(img, (int(1 / 2 * img.shape[1]), int(1 / 2 * img.shape[0])),
                                 interpolation=random.choice([1, 2, 3]))
            # img = np.clip(img, 0.0, 1.0)
            sf = 2

        shuffle_order = random.sample(range(7), 7)
        idx1, idx2 = shuffle_order.index(2), shuffle_order.index(3)
        if idx1 > idx2:  # keep downsample3 last
            shuffle_order[idx1], shuffle_order[idx2] = shuffle_order[idx2], shuffle_order[idx1]

        for i in shuffle_order:

            if i == 0:
                img = self.add_blur(img)

            elif i == 1:
                img = self.add_blur(img)

            elif i == 2:
                a, b = img.shape[1], img.shape[0]
                # downsample2 modify 1
                if random.random() < 1:
                    sf1 = random.uniform(1, 2 * sf)
                    img = cv2.resize(img, (int(1 / sf1 * img.shape[1]), int(1 / sf1 * img.shape[0])),
                                     interpolation=random.choice([1, 2, 3]))

                # img = np.clip(img, 0.0, 1.0)

            elif i == 3:
                # downsample3
                img = cv2.resize(img, (int(1 / sf * a), int(1 / sf * b)), interpolation=random.choice([1, 2, 3]))
                # img = np.clip(img, 0.0, 1.0)

            elif i == 4:
                # add Gaussian noise
                img = self.add_gaussian_noise(img)

            elif i == 5:
                # add JPEG noise
                if random.random() < jpeg_prob:
                    img = self.add_JPEG_noise(img)

        # add final JPEG compression noise
        img = self.add_JPEG_noise(img)

        # random crop
        # img, hq = random_crop(img, hq, sf_ori, lq_patchsize)

        return img, hq


if __name__ == '__main__':
    print(123123)

