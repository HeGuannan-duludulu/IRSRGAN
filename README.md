
# IRSRGAN-Infrared Image Super Resolution GAN
![Github licence](http://img.shields.io/badge/license-MIT-blue.svg)
In order to meet the high definition requirement of important tasks in infrared image application,
I study a method of infrared image super-resolution reconstruction

We propose a model IRSRGAN for infrared images based on super-resolution generative adversarial network (SRGAN). the network structure and downsampling degradation model as well as the model fusion strategy are investigated in depth and trained using a specialized CVC infrared image dataset.

## Main Contribution

### Comprehensive Residual Dense Block(CRDB)




![RRDBVSCRDB.svg](readme_pic%2FRRDBVSCRDB.svg)

**<center>RRDB(a) vs CRDB(b)(Ours)</center>**

--------------

**<center>CRDB reconstruction comparative experiment</center>**
![exp3 crdb.svg](readme_pic%2Fexp3%20crdb.svg)

We improved super-resolution for black-and-white images by addressing interference from a VGG19 model trained on color images. This interference caused green stripe artifacts in black-and-white reconstructions. To solve this, we modified the RRDB structure of ESRGAN and introduced a Comprehensive Residual Dense Block (CRDB) using a global dense residual block. This tailored approach avoids VGG's interference and leverages black-and-white image characteristics, enhancing model performance and stability. Experimental results (see Fig.) confirm the effectiveness of our approach, eliminating green stripe artifacts and achieving good performance in black-and-white image super-resolution.

-------------

**<center>Comparison of SSIM boxplots with and without stochastic degradation strategy (left)</center>**
![EXP1_ssim.svg](readme_pic%2FEXP1_ssim.svg)

This experiment compared two scenarios for an image super-resolution algorithm: one without random degradation and one with random degradation. We used the TISR dataset with 100 randomly preprocessed low-resolution images and measured performance with PSNR and SSIM. We trained models with and without random degradation preprocessing. Results showed that the model with stochastic degradation during training had higher average PSNR and SSIM scores, indicating improved stability and performance for super-resolution with random blur in infrared images.

----------------
**<center>Comparison of reconstruction without randomized degradation and with randomized degradation data enhancement</center>**
![exp1_ran_noran.svg](readme_pic%2Fexp1_ran_noran.svg)

The image reconstruction results of both models have some distortion compared to the original image, but the algorithm with added stochastic degradation is superior in contrast and clarity compared to the other algorithm. Its super-resolution reconstructed image has richer details and textures. And it can reconstruct higher sharpness in the case of noise filled. In summary, this study found that the image super-resolution reconstruction model based on IRSRGAN algorithm performs relatively better in the reconstruction of blurred images in the face of blurring with the addition of stochastic degradation.

----------

### Random Degradation
Our proposed degradation model, which can generate different LR images with a wide range of degradation by different degradation order and degradation methods. In the process of data preprocessing-data enhancement, we set the probability of the process of loading random degradation to 0.25, while the probability of performing normal bilinear downsampling is 0.75. Our proposed new degradation model can better reflect the image degradation in the real world and can provide more degradation samples for the super-resolution reconstruction.

![random degradation.svg](readme_pic%2FCopy%20of%20random%20degradation.svg)

---------------
### LPIPS-I

We also improve and designs a quantitative metric LPIPS-I to measure the quality of IR image reconstruction based on the characteristics of IR images.
Unlike traditional LPIPS metrics, LPIPS-I uses a specific clarity weighting factor to emphasize different image features when measuring the perceived distance between images. This selection of weighting factors is based on the study of cognitive models of image perception by the human visual system and takes into account the influence of the characteristics, texture, and sharpness of infrared images. For infrared images, we use the variance of the Laplacian operator to assess the degree of blurring-i.e., we use the Laplacian operator to do convolution with the input image and then compute the variance, which is used as the clarity weighting factor for the LPIPS-I.


![LPIPS-I.png](readme_pic%2FLPIPS-I.png)

---------------
### Results

![compare_all_imgcompare_all_img (3).svg](readme_pic%2Fcompare_all_imgcompare_all_img%20%283%29.svg)

IRCRRB and IRSRGAN models show excellent performance in image reconstruction.

----------------
### Net Interpolation

We applied a simple linear fusion method to the balancing scheme. The fusion parameter λ was chosen between 0 and 1, with a starting λ = 0.1 and a growth interval of 0.2.

![newnet_interp (2).svg](readme_pic%2Fnewnet_interp%20%282%29.svg)

Purely gan-based methods produce sharp edges and richer textures but with some unwanted artifacts, while purely psnr-oriented methods output blurry images with high psnr values. By using network interpolation, the artifacts are reduced while maintaining the texture. Our network interpolation strategy provides a fusion model that balances perceptual quality and high fidelity, and outputs high-quality reconstructed IR images with both rich texture and high psnr values.



---------------------




    
## Features

- Created a global CRDB residual dense block to enhance fine textures and eliminate green artifacts in reconstructed images.
- Using Pre-trained model to improve the PSNR and SSIM metrics of the images.
- Designed LPIPS-I metrics to measure the quality of IR image reconstruction.
- Designed stochastic degradation strategy to improve the model's adaptability to real-world noise.
- Borrowed the idea of model fusion for better image reconstruction quality.

--------------------------

## Run Locally

Clone the project

```bash
  git clone https://github.com/HeGuannan-duludulu/IRSRGAN
```

Go to the project directory

```bash
  cd IRSRGAN
```

Install dependencies

```bash
   pip install -r requirements.txt
```

Run the Models
```bash
   python model_api.py
```

--------------

## Download Weights


- [Google Driver](https://drive.google.com/file/d/1L6Ev-hHYvJXEaVwjfrILwIehgpa7DVVe/view?usp=drive_link)


---------------------

## How to train and test the model

Train 
- Modify `irsrgan_config.py`file, in line 31, let `mode="train"`
- In line 55, set training epochs `epochs = `
- `python train_irsrgan.py`

Test
- Modify `irsrgan_config.py`file, in line 31, let `mode="test"`
- Modify `model_api.py`file, set `g_model_weights_path = "your_model_path"`
- Run `python model_api.py`

Net_interp
- Modify `net_interp.py`, set `net_PSNR_path`, `net_ESRGAN_path`, `net_interp_path`
- Set `Lambda` to merge different models.

-------------------------


## Thanks

Thanks to @Lornatang . Some code snippets were adapted from [SRGAN-pytorch](https://github.com/Lornatang/SRGAN-PyTorch) project.







