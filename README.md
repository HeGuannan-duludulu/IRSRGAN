
# IRSRGAN-Infrared Image Super Resolution GAN

In order to meet the high definition requirement of important tasks in infrared image application,
I study a method of infrared image super-resolution reconstruction

We propose a model IRSRGAN for infrared images based on super-resolution generative adversarial network (SRGAN). the network structure and downsampling degradation model as well as the model fusion strategy are investigated in depth and trained using a specialized CVC infrared image dataset.

![CRDB.svg](readme_pic%2FCRDB.svg)

![img.png](readme_pic%2Fimg.png)

![EXP1_ssim.svg](readme_pic%2FEXP1_ssim.svg)

![exp1_ran_noran.svg](readme_pic%2Fexp1_ran_noran.svg)

![random degradation.svg](readme_pic%2FCopy%20of%20random%20degradation.svg)

![newnet_interp (2).svg](readme_pic%2Fnewnet_interp%20%282%29.svg)

![LPIPS-I.png](readme_pic%2FLPIPS-I.png)

![compare_all_imgcompare_all_img (3).svg](readme_pic%2Fcompare_all_imgcompare_all_img%20%283%29.svg)
## Acknowledgements

 - [Awesome Readme Templates](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)
 - [Awesome README](https://github.com/matiassingers/awesome-readme)
 - [How to write a Good readme](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)


![Logo](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/th5xamgrr6se0x5ro4g6.png)


## Screenshots

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


## Authors

- [@octokatherine](https://www.github.com/octokatherine)


## Deployment

To deploy this project run

```bash
  npm run deploy
```



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


## Usage/Examples

```javascript
import Component from 'my-project'

function App() {
  return <Component />
}
```


```bash
  npm run test
```


## Installation


    
## Features

- Created a global CRDB residual dense block to enhance fine textures and eliminate green artifacts in reconstructed images.
- Using Pre-trained model to improve the PSNR and SSIM metrics of the images.
- Designed LPIPS-I metrics to measure the quality of IR image reconstruction.
- Designed stochastic degradation strategy to improve the model's adaptability to real-world noise.
- Borrowed the idea of model fusion for better image reconstruction quality.




## Thanks

Thanks to @Lornatang . Some code snippets were adapted from [SRGAN-pytorch](https://github.com/Lornatang/SRGAN-PyTorch) project.


## Contributing

Contributions are always welcome!

See `contributing.md` for ways to get started.

Please adhere to this project's `code of conduct`.




## How to train the model

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
