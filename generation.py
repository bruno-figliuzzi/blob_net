# -*- Encoding: Latin-1 -*-
#!/usr/bin/python

"""
Generation of synthetic images used to train the neural network
"""

import os
from math import *
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.io import imsave


def generate_disk(R):

    """
    Generate an array containing a disk with radius R

    Parameters
    ----------
    R: int
      specified radius

    Returns
    -------
    out: numpy array
       array containing a disk
    """

    x = np.linspace(-R, R, 2*R)
    y = np.linspace(-R, R, 2*R)
    xx, yy = np.meshgrid(x, y)
    output = np.power(xx, 2) + np.power(yy, 2) <= pow(R, 2)
    return output.astype('int')

#--------------------------------------------------
# Image generation
# -------------------------------------------------

class Simulation:

    """
    Generation of synthetic images to train the neural network

    Attributes
    ----------
    nx, ny: int
       size of the simulation domain
    vol: int
       volume of the simulation domain
    xmean: int
       average coordinates at w
    N: int
       number of implanted particles
    x, y: numpy arrays
       coordinates of the implanted particles
    R: numpy array
       sampled radii for the implanted particles
    img: numpy array
       simulated image
    """

    def __init__(self, field, xmean, xscale):

        """
        Class contructor

        Parameters
        -----------
        field: tuple of integers
           simulation field
        xmean: int
           average coordinate at which particles appear
        xscale: int
           corresponding standard deviation
        border: numpy array
           mask for the image border
        mask: numpy array
           mask for the particles
        """

        self.nx, self.ny = field
        self.vol = self.nx*self.ny
        self.xmean = xmean
        self.xscale= xscale


    def generate_borders(self, thickness, amp, rmin, rmax):

        """
        Generate random borders

        Parameters
        ----------
        thickness: float
           thickness of the border
        amp: float
           amplitude of the large scale perturbation
        rmin/rmax: float
           minimal/maximal attenuation factor
        """

        phase = (2*pi/self.nx)*np.arange(self.nx) + np.pi
        bd = (thickness + amp*np.cos(phase)).astype('int')

        self.border = np.ones((self.nx, self.ny))
        lambd = 0.3
        lambd_min, lambd_max = 0.02, 0.8
        for x in range(self.nx):
            for y in range(bd[x]):
                lambd += np.random.normal(loc=0, scale=0.05)
                lambd = min(max(lambd, lambd_min), lambd_max)
                v = rmin + (rmax - rmin)/(1. + exp(-lambd*(y - bd[x])))
                self.border[x, y] = v
                self.border[x, -y] = v


    def simulate(self, theta, R, sigma):

        """
        Simulate particles

        Parameters
        ----------
        theta: float
           average number of particles per surface unit
        R: float
           mean particle radius
        sigma: float
           corresponding standard deviation
        """

        self.N = np.random.poisson(theta*self.vol)
        self.R = np.random.normal(loc=R, scale=sigma, size=self.N).astype('int')
        self.x = np.random.uniform(np.max(self.R), self.nx - np.max(self.R), 
          self.N).astype('int')
        self.y = np.random.uniform(np.max(self.R), self.ny - np.max(self.R), 
          self.N).astype('int')

        keep = []
        for idx, (xc, yc) in enumerate(zip(self.x, self.y)):
            if(self.border[xc, yc] > 0.5 and xc > np.random.normal(
             loc=self.xmean, scale=self.xscale)):
                keep.append(idx)

        self.x = self.x[keep]
        self.y = self.y[keep]
        self.R = self.R[keep]

        # Hardcore process
        idx = 0
        while(idx < self.x.size):

            xc, yc, Rc = self.x[idx], self.y[idx], self.R[idx]

            dist = np.sqrt(np.power(self.x[:idx] - xc, 2) +\
              np.power(self.y[:idx] - yc, 2)) -\
              0.9*(Rc + self.R[:idx])

            keep = (np.sum(dist<0) == 0)
            if(keep):
                idx += 1

            else:
                self.x = np.delete(self.x, idx)
                self.y = np.delete(self.y, idx)
                self.R = np.delete(self.R, idx)
        
        self.N = self.x.size
       

    def generate_mask(self):

        """
        Generate a mask for the particles
        """

        self.mask = np.zeros((self.nx, self.ny))
        for n in range(self.N):

            x, y, R = self.x[n], self.y[n], self.R[n]
            arr = generate_disk(R)
            self.mask[x-R:x+R, y-R:y+R] = np.maximum(
              self.mask[x-R:x+R, y-R:y+R], arr)

        self.mask = self.mask.astype('bool')

    
    def generate_list(self):
        
        """
        Generate a list of the particle center coordinates.
        """

        self.center_coordinates = np.array([self.y,self.x]).T


    def generate_img(self, gray, freqs, amps):

        """
        Generate the final image

        Parameters
        -----------

        gray: int
           average gray level
        freqs: tuple of floats
           frequencies of the periodic noise
        amps: tuple of floats
           amplitudes of the periodic noise     
        """

        f1, f2 = freqs
        amp1, amp2 = amps
        
        # Initialization
        self.img = gray*np.ones((self.nx, self.ny))

        # Periodic noise
        x = np.linspace(0, 1., self.nx) + np.random.normal(loc=0, 
          scale=10./self.nx, size=self.nx)

        y = np.linspace(0, 1., self.ny) + np.random.normal(loc=0, 
          scale=10./self.ny, size=self.ny)

        yy, xx = np.meshgrid(y, x)
        amp1 = amp1*np.ones_like(self.img)
        amp2 = amp2*np.ones_like(self.img)
        amp1[xx < self.xmean/self.nx] /= 4.
        amp2[xx < self.xmean/self.nx] /= 4.

        self.img += amp1*np.cos(2*pi*(f1*xx)) + amp2*np.cos(2*pi*(f2*xx))

        # Particles
        self.particles = 255*np.ones((self.nx, self.ny))

        for n in range(self.N):

            x, y, R = self.x[n], self.y[n], self.R[n]
            arr = generate_disk(R)
            color = np.random.uniform(20, 35)
            arr = 255*(1 - arr) + color*arr
            self.particles[x-R:x+R, y-R:y+R] = np.minimum(
              self.particles[x-R:x+R, y-R:y+R], arr)

        self.img = self.img.ravel()
        self.img[self.mask.astype('bool').ravel()] =\
         self.particles.ravel()[self.mask.astype('bool').ravel()]

        # Add blur and noise, handles borders
        self.img = self.img.reshape((self.nx, self.ny))
        self.img = self.img*self.border 
        self.img = gaussian(self.img, 3)
        noise = np.random.normal(loc=0, scale=8, size=(self.nx, self.ny))
        self.img += noise
        self.img = np.maximum(self.img, 0)


    def add_form(self, v, deg=3):

        """
        Add large scale variations to the image

        Parameters
        ----------

        v: numpy array
           array of relative values for the image intensity
        deg: int
           degree of the fitting polynomial
        """

        c = np.linspace(0, self.nx - 1, len(v))
        p = np.polyfit(c, v, deg)
        xfull = np.arange(self.nx)
        ratio = np.zeros((self.nx, 1))
        for n in range(deg + 1):
            ratio[:, 0] += p[n]*np.power(xfull, deg - n)
        self.img *= np.repeat(ratio, self.ny, axis=1)
        self.img = self.img.astype('uint8')

        
    def create_gaussian_mask(self, std=5):
        """
        Creates a mask with 2d gaussians placed at each center of the particles.
        Gaussians range from 0 to 1 with std="std"
        The image is saved at the specified location 

        Parameters
        ----------
        std: float
            scale of the 2d gaussians    
        """
        size = 20
        
        def gaussian_mask(y0, x0, size, sigma):

            mask_temp=np.zeros(self.img.T.shape) 
            if x0<=size or y0<=size or self.img.T.shape[0]-y0<=size or\
             self.img.T.shape[1]-x0<=size:
                size=int(min(x0,y0,self.img.T.shape[0]-y0,
                 self.img.T.shape[1]-x0))-1
                
            lin = np.arange(-size,size+1)
            x,y = np.meshgrid(lin,lin)
            mask_temp[int(y0)-size:int(y0)+size+1,int(x0)-size:int(x0)+size+1]=\
             np.exp(-(x**2 + y**2)/(2*sigma**2))
            return mask_temp
        
        coord_list = self.center_coordinates
        
        #draw gaussians
        mask=np.zeros(self.img.T.shape)
        for center_coord in coord_list:
            y0, x0 = center_coord
            mask=np.fmax(mask,gaussian_mask(y0,x0,size,std))
        
        #Gaussian filter
        mask=gaussian(mask,sigma=2)

        #save the results
        imsave(os.path.join(labels_dir,'gaussian_'+str(idx)+'.png'), 
         mask, plugin=None, check_contrast=False)
        
# ----------------------------------------------------
# Image simulation
# ---------------------------------------------------- 

def simulate():

    """
    Script for generating a simulated image.

    Returns
    -------
    out: instance of the class Simulation
       simulated image
    """

    field = (4000, 1000)
    xmean = int(np.random.uniform(0, 1500))
    xscale = int(np.random.uniform(0, 30))
    inst = Simulation(field, xmean, xscale)

    # Generate borders
    thickness = int(np.random.uniform(70, 90))
    amp = int(np.random.uniform(7, 10))
    rmin, rmax = 0.1, 1.
    inst.generate_borders(thickness, amp, rmin, rmax)

    # Generate particles
    theta = np.random.uniform(5e-3, 8e-3)
    R = int(np.random.uniform(9, 15))
    sigma = int(np.random.uniform(1, 2))
    inst.simulate(theta, R, sigma)
    inst.generate_mask()

    # Generate image
    gray = int(np.random.uniform(60, 100))
    freqs = np.random.uniform(100, 200, 2)
    amps = np.random.uniform(10, 30, 2)
    inst.generate_img(gray, freqs, amps)
    
    #generate list of coordinates
    inst.generate_list()
    
    #Generate Gaussian mask
    inst.create_gaussian_mask()
    
    # Add large scale variations to the image
    v = np.array([np.random.uniform(0.5, 0.9), 1., 1., 
       np.random.uniform(0.2, 0.8)])

    inst.add_form(v)
    return inst


# ----------------------------------------------------
# Launching script
# ----------------------------------------------------     
if __name__ == '__main__':

    # Specify the location of the folders storing the 
    THIS_FOLDER=os.path.dirname(os.path.abspath(__file__))
    labels_dir = os.path.join(THIS_FOLDER,'dataset/labels')
    train_dir = os.path.join(THIS_FOLDER,'dataset/train')
    val_dir = os.path.join(THIS_FOLDER,'dataset/val')

    # Training set
    for idx in range(0,160):
        print(idx)
        inst = simulate()
        imsave(os.path.join(train_dir,str(idx) + ".png"), inst.img.T)
        np.save(os.path.join(labels_dir,"binary_" + str(idx) + ".npy"), 
         inst.mask.T)

    # Validation set
    for idx in range(160,200):
        print(idx)
        inst = simulate()
        imsave(os.path.join(val_dir,"img" + str(idx) + ".png"), inst.img.T)
        np.save(os.path.join(labels_dir,"binary_" + str(idx) + ".npy"), 
         inst.mask.T)
         
