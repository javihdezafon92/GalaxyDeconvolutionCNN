"""
In this file we collect the set of functions used to display visual results and to save them
"""


import os

# Math and data
import numpy as np
import random
import pandas as pd

# Image processing and plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import cv2

# Astropy
from astropy.io import fits
from astropy.visualization import astropy_mpl_style

# Local functions
from TFMPackage import architectures_V2
from astropy.visualization import astropy_mpl_style
from sklearn.model_selection import train_test_split
from TFMPackage import train_utils

# Machine learning
import sklearn as sk
from skimage.metrics import structural_similarity as ssim


def display(array1, array2, n = 10):
    """
    Displays n random images from each one of the supplied arrays.
    
    :param ndarray array1: array 1 of image dataset
    :param ndarray array2: array 2 of image dataset
    :param int n: number of images displayed    
    """

    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]
    
    plt.figure(figsize=(20, (40/n)))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(64, 64))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(64, 64))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()
   
def results_comparison(X_test, y_test, maxim, predictions_liu, predictions_own, r = 255, index = 2):
    """
    Compares the deconvolution result obtained by liu and lam model and our model, of the galaxy indicated in index
    
    :param ndarray X_test: set of galaxies affected by aberrations
    :param ndarray y_test: set of original galaxies
    :param float maxim: maximum intensity value of the set of images
    :param ndarray predictions_liu: predicted images by Liu and Lam model
    :param ndarray predictions_own: predicted images by our model
    :param int index: index of the galaxy to be compared
    
    """
    figure = plt.gcf()
    figure.set_size_inches(15,5)   

    PSNR_noise = cv2.PSNR(X_test[index,:,:,0].astype("float32")*r, y_test[index,:,:,0].astype("float32")*r,r)
    PSNR_liu = cv2.PSNR(predictions_liu [index,:,:,0].astype("float32")*r, y_test[index,:,:,0].astype("float32")*r,r)
    PSNR_own = cv2.PSNR(predictions_own[index,:,:,0].astype("float32")*r, y_test[index,:,:,0].astype("float32")*r,r)

    SSIM_noise = ssim(X_test[index,:,:,0], y_test[index,:,:,0], data_range=y_test[index,:,:,0].max() - y_test[index,:,:,0].min())
    SSIM_liu = ssim(predictions_liu[index,:,:,0], y_test[index,:,:,0], data_range=y_test[index,:,:,0].max() - y_test[index,:,:,0].min())
    SSIM_own = ssim(predictions_own[index,:,:,0], y_test[index,:,:,0], data_range=y_test[index,:,:,0].max() - y_test[index,:,:,0].min())

    plt.subplot(141)                
    plt.title('Ground truth')
    plt.imshow(y_test[index,:,:,0]*r)
    plt.axis('off')

    plt.subplot(142)                
    plt.title('With anomalies \n PSNR = ' + str(round(PSNR_noise,2)) + ' dB | SSIM = ' + str(round(SSIM_noise,2)))
    plt.imshow(X_test[index,:,:,0]*r)
    plt.axis('off')

    plt.subplot(143)                
    plt.title('Liu and Lam prediction \n PSNR = ' + str(round(PSNR_liu,2)) + ' dB | SSIM = ' + str(round(SSIM_liu,2)))
    plt.imshow(predictions_liu[index,:,:,0]*r)
    plt.axis('off')

    plt.subplot(144)                
    plt.title('Our model prediction \n PSNR = ' + str(round(PSNR_own,2)) + ' dB | SSIM = ' + str(round(SSIM_own,2)))
    plt.imshow(predictions_own[index,:,:,0]*r)
    plt.axis('off')
    
def results_difference(X_test, y_test, maxim, predictions_own, r = 255, index = 2):
    
    '''
    Obtains the image difference between an image afected by anomalies and the output image of our model
    
    :param ndarray X_test: set of galaxies affected by aberrations
    :param ndarray y_test: set of original galaxies
    :param float maxim: maximum intensity value of the set of images    
    :param ndarray predictions_own: predicted images by our model
    :param int r: maximum intensity value
    :param int index: index of the galaxy to be compared
    '''
    
    figure = plt.gcf()              
    figure.set_size_inches(15,5)    

    plt.subplot(141)                
    plt.title('Ground truth')
    plt.imshow(y_test[index,:,:,0] * r, vmin = 0, vmax = r)
    plt.axis('off')

    plt.subplot(142)                
    plt.title('With anomalies')
    plt.imshow(X_test[index,:,:,0] * r, vmin = 0, vmax = r)
    plt.axis('off')

    plt.subplot(143)                
    plt.title('Prediction')
    plt.imshow(predictions_own[index,:,:,0] * r, vmin = 0, vmax = r)
    plt.axis('off')

    plt.subplot(144)                
    plt.title('Difference \n prediction - ground_truth')
    plt.imshow(abs(y_test[index,:,:,0] - predictions_own[index,:,:,0]) * r, vmin = 0, vmax = r)
    plt.axis('off')
   
def get_fits(X_test, y_test, maxim, predictions_liu, predictions_own, PSF = False):
    
    '''
    Generates the .fits files of the original image, the one afected by anomalies, the liu and lam output and our model output of three random galaxies
    
    :param ndarray X_test: set of galaxies affected by aberrations
    :param ndarray y_test: set of original galaxies
    :param float maxim: maximum intensity value of the set of images    
    :param ndarray predictions_own: predicted images by our model
    :param ndarray predictions_liu: predicted images by Liu and Lam model
    :param bool PSF: indicates problem type
    '''

    if PSF == True:
        problem_type = 'PSF'
    else:
        problem_type = 'Denoising'
    
    current_dir=os.getcwd()
    save_dir = train_utils.create_directory_images(problem_type = problem_type)

    index = random.randint(0, len(y_test) - 3)

    for i in range(index, index + 2):   

        hdu_original = fits.PrimaryHDU(y_test[i,:,:,0] * maxim)
        hdul_original = fits.HDUList([hdu_original])    
        path_original = save_dir + "\\Original_" + str(i) + ".fits"
        hdul_original.writeto(path_original)

        hdu_ruido = fits.PrimaryHDU(X_test[i,:,:,0] * maxim)
        hdul_ruido = fits.HDUList([hdu_ruido])    
        path_ruido = save_dir + "\\Anomalies_" + str(i) + ".fits"
        hdul_ruido.writeto(path_ruido)    

        hdu_limpia_myModel = fits.PrimaryHDU(predictions_own[i,:,:,0] * maxim)
        hdul_limpia_myModel = fits.HDUList([hdu_limpia_myModel])
        path_limpia_myModel = save_dir + "\\ourModel_Prediction" + str(i) + ".fits"
        hdul_limpia_myModel.writeto(path_limpia_myModel)

        hdu_limpia_LiuLam = fits.PrimaryHDU(predictions_liu[i,:,:,0] * maxim)
        hdul_limpia_LiuLam = fits.HDUList([hdu_limpia_LiuLam])    
        path_limpia_LiuLam  = save_dir + "\\liuLam_Prediction" + str(i) + ".fits"
        hdul_limpia_LiuLam.writeto(path_limpia_LiuLam)