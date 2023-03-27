
# Introduction

This repository contains the code developed for my final project for the master's degree in artificial intelligence at the International University of La Rioja. It consisted in the development of a deep learning model, based on convolutional neural networks, for the correction of optical aberrations in galaxy images.

In astronomical image acquisition, it is common to find artifacts and anomalies because the particularities of the studied objects (light intensity, physical nature, etc.) as well as the acquisition process (instrumental aberrations, atmospheric turbulunce, etc.). Two of these aberrations are the Poisson noise and the effect of the  point spread function (PSF).

Poisson noise occurs due to the oscillatory nature of light measurements by optical captation instruments. The low number of photons that the instruments capture means that this noise can be modeled using a Poisson distribution. It has the particularity of being closely correlated with the real image. On the other hand, the PSF models the response of an optical captation system to an input in the form of a Dirac delta, and it generates a blurring effect and a loss of spatial resolution. In the case of shift invariant systems, the resulting image can be approximated as the convolution of the real image with the PSF. 

Motivated for the recent advances in the field of Deep Learning for image reconstruction, we have built a solution based on convolutional neural networks (for astronomic image aberrations removal). We have built the model based on the framework described by Mao, Shen and Yang (2016) for the construction of CNNs specialised in image restoration, using architectures formed by auto encoders and skip connections. We have also based our model from the one proposed by Liu Lam (2018) with a proven efficiency for the removal of Poisson noise in natural images (which is also based on the text of Mao et al.). 

For the full deconvolution problem, we can increase the spatial resolution of the images, recovering the core and the main structure of thegalaxy, with gains between 7 and 8 dB for PSNR and around 0.5 for SSIM. However, the model is not able to recover weaker structures associated with the halo and point like objects. Although a good partial reconstruction of the galaxy is achieved, there is information of interest that is lost.


# Contents

## Package

- *architectures_V2*: contains the classes that define the Liu and Lam's model architecture, as well as all the variations made following de Shen and Yang framework recommendations, including our final model, making use of the Keras functions.
- *train_utils*: contains all the function used to train and evaluate the models.
- *plot_utils*: contains the functions realises to visualise results and store images generated by the models.

## Notebooks
- *Multiple Trainnings*: here we train all the Liu and Lam model and all the architecture variations realized and we compare the PSNR and SSIM results obtaindes. Firstly, we train the models for the denoising problem only, and then for the full deconvolution problem.
- *Final Model Trainning*: here we only train the Liu and Lam model and our final model and we compare the results obtained in the same way we did in the *Multiple Trainnings* notebook.
- *Plotting*: here we show the visual results obtained by the trainned models.

## Data

***IMPORTANT**: Due to the large size of the dataset used, it is not uploaded in the repository. As a temporary solution, you can download it from the following dropbox link: https://www.dropbox.com/s/utfur8x24ij7r8u/data.rar?dl=0*

The required dataset to train and evaluate the model have been artificially generated using the GalSim software (Rowe et al., 2015). We generate a total of 8.100 galaxies affected by the aberrations and their respective ground truth, and we dedicate 70% of the datasets for training, 20% for validation and 10% for
testing. As evaluation metrics, we use the peak signal to noise ratio (PSNR) and the structural similarity (SSIM).

You will find 5 files in the dataset folder.

- *multi_noPSF_noNoise*: it contains the galaxies generated unaffected by noise and the psf effect. These images serve as ground truth during training.

- *multi_PSF_noise_L4, multi_PSF_noise_L5, multi_PSF_noise_L6*: they contain the galaxies affected only by noise (three noise levels, increasing from L4 to L6). These images are the inputs of the models trained only for the denoising problem.

- *multi_PSF_Noise*: is the dataset of galaxies affected by noise and by the PSF effect. These images are the inputs of the models trained for the full deconvolution problem.


# Get more information

For more information you can consult the project report in the following link: https://www.dropbox.com/s/aj8z8u7lvonvcrd/Project_Report_Javier_Hern%C3%A1ndez_Afonso.pdf?dl=0

You can also view a poster presented in the XV Scientific Meeting of the Spanish Astronomical Society (SEA): https://doi.org/10.5281/zenodo.7048828


# References

[1]Liu, P. Y. Lam, E. Y. (2018). Image Reconstruction Using Deep Learning. https://arxiv.org/abs/1809.10410v1

[2]Mao, X., Shen, C., & Yang, Y. B. (2016). Image restoration using very deep convolutional encoder-decoder networks with symmetric skip connections. Advances in neural information processing systems, 29. https://arxiv.org/abs/1603.09056

[3]Rowe, B. T., Jarvis, M., Mandelbaum, R., Bernstein, G. M., Bosch, J., Simet, M., ... & Gill, M. S. (2015). GALSIM: The modular galaxy image simulation toolkit. Astronomy and Computing, 10, 121-150. https://www.sciencedirect.com/science/article/abs/pii/S221313371500013X
