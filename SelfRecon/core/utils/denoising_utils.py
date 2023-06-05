import numpy as np
import os
from .common_utils import *
import cv2


def get_noisy_image(img_np, sigma, scale=0):
    """Adds Gaussian noise to an image.

    Args:
        img_np: image, np.array with values from 0 to 1
        sigma: std of the noise
    """
    if scale != 0:
       print(f"{scale} Poisson noises are applied.") # what would these individual pixels intensities be if they were produced by a Poisson process
       img_noisy_np = np.clip(np.random.poisson(lam=img_np/scale, size=img_np.shape)*scale, 0, 1).astype(np.float32)
    else:
       print(f"(Sigma={sigma}) Gaussian noises are added.")
       img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)

    return img_noisy_np

def distance(point1, point2):
  return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def gaussian_LP(D0, imgShape):
  base = np.zeros(imgShape[-2:])
  rows, cols = imgShape[-2:]
  center = (rows/2, cols/2)
  for x in range(cols):
     for y in range(rows):
       base[y,x] = np.exp(((-distance((y,x),center)**2)/(2*(D0**2))))
  return base

def plain_LP(D0, imgShape):
  base = np.zeros(imgShape[-2:])
  rows, cols = imgShape[-2:]
  center = (rows/2, cols/2)
  for x in range(cols):
    for y in range(rows):
      base[y,x] = distance((y,x), center) < D0
  return base

def getBlurred(img, r=16, type='gauss'):
  if type == 'gauss':
     kernel = gaussian_LP(r, img.shape)
  else:
     kernel = plain_LP(r, img.shape)

  kernel = np.broadcast_to(kernel, (3,512,512)).transpose(1,2,0)
  img = img.transpose(1,2,0) * 255
  b,g,r = cv2.split(img)
  freq = np.fft.fftshift(np.fft.fft2(img))
  freq_masked = np.fft.fftshift(freq * kernel)
  img_lp_mag = np.abs(np.fft.ifft2(freq_masked)).real
  img_lp = cv2.normalize(img_lp_mag,None,0,255,cv2.NORM_MINMAX, cv2.CV_8U)
  b,g,r = map(img_lp, (b,g,r))
  img_lp = cv2.merge((b,g,r))
  return img_lp     
 
