# -*- Encoding: Latin-1 -*-
#!/usr/bin/python

"""
Segmentation with a classical, non-supervised approach
"""

import os
import numpy as np
from matplotlib import pyplot as plt

from skimage.io import imread
from skimage.morphology import binary_opening, disk, erosion
from skimage.segmentation import mark_boundaries, watershed
from skimage.measure import label
from skimage.feature import peak_local_max
from sklearn.cluster import KMeans
from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_otsu, threshold_local



class Segmentation:

    """
    Particles segmentation with a classical, non-supervised approach

    Attributes
    ----------

    img: numpy array
       original image
    w, h: int
       width/height of the image
    mask: numpy array
       binary mask for the particles
    coordinates: numpy array
       coordinates of the particles centers
    """

    def __init__(self, img):

        """
        Load and pre-process the input image

        Parameters
        ----------

        img: numpy array
           input image
        """

        self.img = img[:, :, 0] 
        self.w, self.h = self.img.shape


    def binarize_kmeans(self):

        """
        Binarize the image using the K-Means algorithm
        """

        n_segments = 2  
        kmeans = KMeans(n_clusters=n_segments)
        kmeans.fit(self.img.ravel().reshape(-1,1))
        labels = kmeans.labels_.reshape((self.w, self.h))

        # Construct the binarized image: the particles are labelled 1, 
        # the background 0
        idx = np.argmin(kmeans.cluster_centers_)
        self.mask = np.zeros((self.w, self.h), dtype='int')
        self.mask[labels==idx] = 1

        # Filter the binary image
        self.mask = binary_opening(self.mask, disk(3))
        
        
    def otsu_threshold(self):

        """
        Binarize the image using Otsu thresholding    
        """
        thresh = threshold_otsu(self.img)
        self.mask = self.img < thresh
        self.mask = binary_opening(self.mask, disk(3))
        
        
    def adaptative_threshold(self, block_size, offset):

        """
        Binarize the image using adaptative thresholding
        
        Parameters
        ----------
        
        block_size: int
           size of the local neighborhood used to compute the threshold
        offset: int
           offset used for the threshold   
        """
        
        local_thresh = threshold_local(self.img, block_size, offset=offset)
        self.mask = self.img < local_thresh
        self.mask = binary_opening(self.mask, disk(3))
            
   
    def extract_centers(self, min_distance):

        """
        Extract the coordinates of the particles centers

        Parameters
        ----------

        min_distance: float
           minimal distance between the particles centers
        """

        # Compute the distance image
        self.img_distance =  distance_transform_edt(self.mask)

        # Extract the peaks of the distance function
        self.coordinates = peak_local_max(self.img_distance, 
          min_distance=min_distance, 
          exclude_border=False)


    def watershed_segmentation(self):

        """
        Watershed segmentation to separate individual particles

        Note
        ----

        The watershed algorithm takes the centers of the particles as inner
        markers.
        """

        # Select internal and external markers
        markers = erosion((self.mask==0), disk(1))
        for (x,y) in self.coordinates:
            markers[x, y] = 1
        segments = watershed(np.max(self.img_distance)-self.img_distance, 
          label(markers))

        # Keep only the regions growing from the internal markers
        self.img_regions = np.zeros_like(segments)
        for idx, (x,y) in enumerate(self.coordinates):
            lb = segments[x, y]
            self.img_regions[segments==lb] = idx


    def display(self):
        
        """
        Display the results:
        - original image with the centers of the particles superimposed
        - binary mask for the particles
        - distance image used to compute the watershed
        - label image of the particles
        """

        fig, ax = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
        img_boundaries = mark_boundaries(self.img, self.img_regions)

        ax[0, 0].imshow(self.img, cmap=plt.cm.gray)
        ax[0, 0].scatter(self.coordinates[:, 1], self.coordinates[:, 0])
        ax[0, 1].imshow(self.mask, cmap=plt.cm.gray)
        ax[1, 0].imshow(self.img_distance)
        ax[1, 1].imshow(self.img_regions)
       
        for a in ax.ravel():
            a.set_axis_off()
        plt.tight_layout()
        plt.show()
        plt.close()


# ----------------------------------------------------
# Launching script
# ----------------------------------------------------     
if __name__ == '__main__':

    path = './dataset/imgs/'
    names = os.listdir(path)
    for name in names:
    
        img = imread(path + name)
        inst = Segmentation(img)
        #inst.binarize_kmeans()
        #inst.adaptative_threshold(81, 0)
        inst.otsu_threshold()
        inst.extract_centers(min_distance=10)
        segments = inst.watershed_segmentation()
        inst.display()
        #np.save('./results/segments_synthesis/' + name[:-4] + '_otsu.npy', 
        # inst.img_regions)

