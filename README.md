
# Introduction

This repository contains the code developed for my final project for the master's degree in artificial intelligence at the International University of La Rioja. It consisted in the development of a deep learning model, based on convolutional neural networks, for the correction of optical aberrations in galaxy images.

In astronomical image acquisition, it is common to find artifacts and anomalies because the particularities of the studied objects ( light intensity, physical nature, etc.) as well as the acquisition process (instrumental aberrations, atmospheric turbulunce, etc.). Two of these aberrations are the Poisson noise and the effect of the  point spread function (PSF).

Poisson noise occurs due to the oscillatory nature of light measurements by optical captation instruments. The low number of photons that the instruments capture means that this noise can be modeled using a Poisson distribution. It has the particularity of being closely correlated with the real image. On the other hand, the PSF models the response of an optical captation system to an input in the form of a Dirac delta, and it generates a blurring effect and a loss of spatial resolution. In the case of shift invariant systems, the resulting image can be approximated as the convolution of the real image with the PSF. 

Motivated for the recent advances in the field of Deep Learning for image reconstruction, we have built a solution based on convolutional neural network (for astronomic image aberrations removal). We have built the model based on the framework described by Mao, Shen and Yang 2016 for the construction of CNNs specialised in image restoration, using architectures formed by auto encoders and skip connections We have also based our model from the one proposed by Liu Lam 2018 with a proven efficiency for the removal of Poisson noise in natural images (which is also based on the text of Mao et al.)

# References

[1]Liu, P. Y. Lam, E. Y. (2018). Image Reconstruction Using Deep Learning https:://arxiv.org/abs/1809.10410v1

[2]Mao, X., Shen, C., & Yang, Y. B. (2016). Image restoration using very deep convolutional encoder-decoder networks with symmetric skip connections. Advances in neural information processing systems, 29. https://arxiv.org/abs/1603.09056
