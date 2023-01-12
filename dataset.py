"""
Methods used by the neural network to handle the dataset
---------------------------------------------------------

Load pairs of image and target from the dataset, and define
transformations on the images (random crop, histogram distortion,
horizontal flip)
"""

import torch
import skimage.transform as tr
from skimage import img_as_float
import os
from skimage import io
from skimage.color import rgb2hsv, hsv2rgb, gray2rgb
import numpy as np

# ------------
# Tools
# ------------

def rand_scale(s):

    """
    Return a random scale

    Parameters
    ----------
    s: float
        scale parameter
   
    Returns
    -------
    out: float
        random scale
    """

    scale = np.random.uniform(1, s)
    if(np.random.randint(1, 10000) % 2): 
        return scale
    return 1./scale


# ------------------
# Dataset handling
# ------------------

class SegmentationDataset:

    """
    Class handling the segmentation dataset

    Attributes
    ----------
    root_dir: string
        root directory 
    input_dir: string
        path to the training images, specified relatively to root_dir
    target_dir: string
        path to the training targets, specified relatively to root_dir
    transform: python list
        list of the transformations to apply to the images
    images: list of string
        list containing the location of each image in the dataset
    """

    def __init__(self, root_dir, input_dir, target_dir, transform = None):

        """
        Class constructor

        Initialization:
        - set the location of the folders containing the dataset as 
          attributes of the class
        - list the images of the dataset 
        - set the transforms to be applied to the images as attributes of
          the class

        Parameters
        ----------
        root_dir: string
            root directory 
        input_dir: string
            path to the training images, specified relatively to root_dir
        target_dir: string
            path to the training targets, specified relatively to root_dir
        transform: python list
            list of the transformations to apply to the images
        """

        self.root_dir = root_dir
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.images = os.listdir(os.path.join(self.root_dir, self.input_dir))


    def __len__(self):

        """
        Return the number of images in the dataset

        Returns
        --------

        out: int
            number of images in the dataset
        """
        return len(self.images)


    def __getitem__(self, idx):

        """
        Return an image of the dataset along with the corresponding
        target images: segmentation mask and target image, where the 
        center of each particle is represented by a Gaussian with fixed 
        scale.

        The specified data augmentation transforms are applied to the returned
        image.

        Parameters
        ----------
        idx: int
            index of the image in the dataset
        
        Returns
        -------
        out: dictionary
            dictionary containing the input image (key: "input"),
            the target images (key: "target" and "target_gauss") and
            the name of the image (key: "name")
        """

        img_name = os.path.join(self.root_dir,
          self.input_dir,
          self.images[idx])

        target_name = os.path.join(self.root_dir,
          self.target_dir,
          str("binary_") + self.images[idx][:-4] + '.npy')
          
        target_gauss_name = os.path.join(self.root_dir,
          self.target_dir,
          str("gaussian_") + self.images[idx][:-4] + '.png')

        img =  gray2rgb(io.imread(img_name))
        target = np.load(target_name, allow_pickle=True)
        target_gauss = io.imread(target_gauss_name)
        
        sample = {'input': img, 'target': target, 'target_gauss': target_gauss,
          'name': self.images[idx][:-4]}

        if self.transform:
            sample = self.transform(sample)

        return sample



class TestSegmentationDataset:

    """
    Class handling the test dataset

    Attributes
    ----------
    root_dir: string
        root directory 
    input_dir: string
        path to the training images, specified relatively to root_dir
    images: list of string
        list containing the location of each image in the dataset
    """

    def __init__(self, root_dir, input_dir):

        """
        Class constructor

        Initialization:
        - set the location of the folders containing the dataset as 
          attributes of the class
        - list the images of the dataset 
        - set the transforms to be applied to the images as attributes of
          the class

        Parameters
        ----------
        root_dir: string
            root directory 
        input_dir: string
            path to the training images, specified relatively to root_dir
        """

        self.root_dir = root_dir
        self.input_dir = input_dir
        self.images = os.listdir(os.path.join(self.root_dir, self.input_dir))


    def __len__(self):

        """
        Return the number of images in the dataset

        Returns
        --------
        out: int
            number of images in the dataset
        """

        return len(self.images)


    def __getitem__(self, idx):

        """
        Return the image of the dataset corresponding to the specified index

        Parameters
        ----------
        idx: int
            index of the image in the dataset
        
        Returns
        -------
        out: dictionary
            dictionary containing the input image (key: "input"),
            and the name of the image (key: "name")
        """

        img_name = os.path.join(self.root_dir,
          self.input_dir,
          self.images[idx])

        #img =  gray2rgb(io.imread(img_name)[:, :, 0])
        img = io.imread(img_name)
        img = img_as_float(img).transpose((2, 1, 0))
        img = torch.from_numpy(img).float()

        sample = {'input': img, 'name': self.images[idx][:-4]}
        return sample



# -------------------------------------------------
# Data augmentation methods
# -------------------------------------------------

class Crop(object):

    """
    Crop images from a sample

    Attributes
    ----------
    output_size: tuple or int
        requested output size. If int, a square crop
        is extracted.
    top: int
        x-coordinate of the upperleft pixel defining the crop
    left: int 
        y-coordinate of the upperleft pixel defining the crop
    """

    def __init__(self, output_size, upperleft = (0,0)):

        """
        Class constructor

        Parameters
        ----------
        output_size: tuple or int
            requested output size. If int, a square crop
             is extracted.
        upperleft: tuple
            coordinates of the upper left pixel defining the crop
        """

        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)

        else:
            assert len(output_size) == 2
            self.output_size = output_size

        self.top = upperleft[1]
        self.left = upperleft[0]


    def __call__(self, sample):

        """
        Apply the transform to the specified sample

        Parameters
        ----------
        sample: dictionary
            dictionary containing the input image (key: "input")
            the target images (key: "target" and "target_gauss") and
            the image name
        
        Returns
        -------
        out: dictionary
            dictionary containing the  transformed input image (key: "input")
            the target images (key: "target" and "target_gauss") and
            the image name
        """

        img, target, target_gauss, name = sample['input'], sample['target'],\
         sample['target_gauss'], sample['name']

        h, w = img.shape[:2]
        new_h, new_w = self.output_size

        img = img[self.top:self.top+new_h, self.left:self.left+new_w]
        target = target[self.top:self.top+new_h, self.left:self.left+new_w]
        target_gauss = target_gauss[self.top:self.top+new_h,
         self.left:self.left+new_w]
        
        return {'input': img, 'target': target,
         'target_gauss': target_gauss, 'name': name}



class RandomCrop(object):

    """
    Randomly crop images from a sample.

    Parameters
    ----------
    output_size: tuple or int
       requested output size. If int, a square crop is extracted.
    """

    def __init__(self, output_size):

        """
        Class constructor

        Parameters
        ----------
        output_size: tuple or int
            requested output size. If int, a square crop
             is extracted.
        """

        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size


    def __call__(self, sample):

        """
        Apply the transform to the specified sample

        Parameters
        ----------
        sample: dictionary
            dictionary containing the input image (key: "input")
            the target images (key: "target" and "target_gauss") and
            the image name
        
        Returns
        -------
        out: dictionary
            dictionary containing the  transformed input image (key: "input")
            the target images (key: "target" and "target_gauss") and
            the image name
        """

        img, target, target_gauss, name = sample['input'], sample['target'],\
         sample['target_gauss'], sample['name']

        h, w = img.shape[:2]
        new_h, new_w = self.output_size
        
        rand = np.random.uniform(0.3,2.0)
        rand_h, rand_w = int(rand*new_h), int(rand*new_w)
        
        top = np.random.randint(0, h - rand_h)
        left = np.random.randint(0, w - rand_w)

        img = tr.resize(img[top:top + rand_h,left:left + rand_w], 
         (new_h, new_w), anti_aliasing=True)
        target = tr.resize(target[top:top + rand_h,left:left + rand_w], 
         (new_h, new_w), anti_aliasing=False,order=0)
        target_gauss = tr.resize(
         target_gauss[top:top + rand_h,left:left + rand_w], 
         (new_h, new_w), anti_aliasing=True)

        return {'input': img, 'target': target,
         'target_gauss': target_gauss, 'name': name}


class Flip(object):

    """
    Flip the image horizontally
    """

    def __call__(self, sample):

        """
        Apply the transform to the specified sample

        Parameters
        ----------
        sample: dictionary

            dictionary containing the input image (key: "input")
            the target images (key: "target" and "target_gauss") and
            the image name
        
        Returns
        -------
        out: dictionary
            dictionary containing the  transformed input image (key: "input"),
            the target images (key: "target" and "target_gauss") and
            the image name
        """

        img, target, target_gauss, name = sample['input'], sample['target'],\
          sample['target_gauss'], sample['name']
        
        flip = np.random.randint(0, 2) 

        if(flip):
            img = np.flip(img,axis=1).copy()
            target = np.flip(target,axis=1).copy()
            target_gauss = np.flip(target_gauss,axis=1).copy()

        """
        Notes on np.flip:
        Current version of torch.from_numpy() does not support
        negative ndarray striding. A -not so elegant- solution
        to this problem, is to make a copy of the ndarray, during which
        process, the negative strides are removed.
        """

        return {'input': img, 'target': target, 'target_gauss': target_gauss,\
         'name': name}  


class Rotate(object):

    """
    Rotates the image by 
	
	.. math::
	n \times 90°
	
	where n is a random integer in [0,3].
    To be used only with square crops, or if batch_size==1
    """

    def __call__(self, sample):

        """
        Apply the transform to the specified sample

        Parameters
        ----------
        sample: dictionary
            dictionary containing the input image (key: "input")
            the target images (key: "target" and "target_gauss") and
            the image name
        
        Returns
        -------
        out: dictionary
            dictionary containing the  transformed input image (key: "input"),
            the target images (key: "target" and "target_gauss") and
            the image name
        """

        img, target, target_gauss, name = sample['input'], sample['target'],\
         sample['target_gauss'], sample['name']
        
        angle = np.random.randint(0, 4) 

        if(angle==1):
            img = tr.rotate(img,90)
            target = tr.rotate(target,90)
            target_gauss = tr.rotate(target_gauss,90)
        elif(angle==2):
            img = tr.rotate(img,180)
            target = tr.rotate(target,180)
            target_gauss = tr.rotate(target_gauss,180)
        elif(angle==3):
            img = tr.rotate(img,270)
            target = tr.rotate(target,270)
            target_gauss = tr.rotate(target_gauss,270)
        else:
            pass
            
        
        return {'input': img, 'target': target, 'target_gauss': target_gauss,\
         'name': name}  


class Distort(object):

    """
    Distort the image histogram
    """

    def __call__(self, sample):

        """
        Apply the transform to the specified sample

        Parameters
        ----------
        sample: dictionary
            dictionary containing the input image (key: "input")
            the target images (key: "target" and "target_gauss") and
            the image name
        
        Returns
        -------
        out: dictionary
            dictionary containing the  transformed input image (key: "input"),
            the target images (key: "target" and "target_gauss") and
            the image name
        """

        img, target, target_gauss, name = sample['input'], sample['target'],\
         sample['target_gauss'], sample['name']
        img_hsv = rgb2hsv(img)

        hue = np.random.uniform(-0.1, 0.1)
        saturation = rand_scale(0.5)
        exposure = rand_scale(0.5)

        img_hsv[:, :, 1] *= saturation
        img_hsv[:, :, 2] *= exposure

        img_hsv[:, :, 0] += 255*hue
        img_hsv[(img_hsv[:, :, 0]>255), 0] -= 255 
        img_hsv[(img_hsv[:, :, 0]<0), 0] += 255  

        img = hsv2rgb(img_hsv)      

        return {'input': img, 'target': target, 'target_gauss': target_gauss,
         'name': name}


class SetTarget(object):

    """
    Construct the target image from the labels
    """

    def __call__(self, sample):

        """
        Apply the transform to the specified sample

        Parameters
        ----------
        sample: dictionary
            dictionary containing the input image (key: "input")
            the target images (key: "target" and "target_gauss") and
            the image name
        
        Returns
        -------
        out: dictionary
            dictionary containing the  transformed input image (key: "input"),
            the target images (key: "target" and "target_gauss") and
            the image name
        """

        img, target, target_gauss, name = sample['input'], sample['target'],\
         sample['target_gauss'], sample['name']
        target = np.stack((target,target_gauss),axis=2)

        return {'input': img, 'target': target,\
         'target_gauss': target_gauss, 'name': name}


class Normalize(object):

    """
    Normalizes images from a sample between 0 and 1
    """

    def __call__(self, sample):

        """
        Apply the transform to the specified sample

        Parameters
        ----------
        sample: dictionary
            dictionary containing the input image (key: "input")
            the target images (key: "target" and "target_gauss") and
            the image name
        
        Returns
        -------
        out: dictionary
            dictionary containing the  transformed input image (key: "input"),
            the target images (key: "target" and "target_gauss") and
            the image name
        """

        img, target, target_gauss, name = sample['input'], sample['target'],\
         sample['target_gauss'], sample['name']
        img = img_as_float(img)
        target = img_as_float(target)
        target_gauss = img_as_float(target_gauss)
        
        return {'input': img, 'target': target,\
         'target_gauss': target_gauss, 'name': name}


class ToTensor(object):

    """
    Converts the ndarrays from the samples into PyTorch Tensors
    """

    def __call__(self, sample):

        """
        Apply the transform to the specified sample

        Parameters
        ----------
        sample: dictionary
            dictionary containing the input image (key: "input")
            the target images (key: "target" and "target_gauss") and
            the image name
        
        Returns
        -------
        out: dictionary
            dictionary containing the  transformed input image (key: "input"),
            the target images (key: "target" and "target_gauss") and
            the image name
        """

        img, target, target_gauss, name = sample['input'], sample['target'],\
         sample['target_gauss'], sample['name']
        
        img = img.transpose((2, 1, 0))
        target = target.transpose((2, 1, 0))

        img = torch.from_numpy(img).float()
        target = torch.from_numpy(target).float()

        return {'input': img, 'target': target,\
         'target_gauss': target_gauss, 'name': name}


