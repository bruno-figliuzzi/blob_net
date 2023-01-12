"""
Image segmentation 
"""

import sys, time, os
from os.path import expanduser
home = expanduser("~")

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

from network import Net
from dataset import *

from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.measure import label
from skimage.morphology import binary_erosion, binary_opening, disk
from skimage.filters.rank import gradient
from skimage.segmentation import watershed



def extract_coordinates(output):

    """
    Extract the coordinates of the particles and the segmentation mask
    from the output of the convolutional network

    Parameters
    ----------

    output: pytorch tensor
       output of the neural network
    
    Returns
    ------- 
    out: numpy arrays
       segmentation mask and coordinates of the particles:
       - the segmentation mask is represented by a binary mask with the same
         size as the original image
       - the coordinates are represented by a three columns array containing
         the x-coordinate, the y-coordinate, the intensity at the Gaussian 
         location
    """

    output = (255*output.data.cpu().numpy()[0]).astype('int') 
    output = np.transpose(output, (2, 1, 0) )

    mask = binary_opening(output[...,0]>230, disk(3))
    
    GaussMask = gaussian(output[...,1], sigma = 2)
    GaussMask = ( GaussMask / np.max(GaussMask) ) * 255
    coords = peak_local_max(mask*GaussMask,min_distance=5, threshold_abs=0.1)
    return mask, coords


def apply_watershed(mask, coords):

    """
    Apply a watershed segmentation to the mask/markers outputed by the 
    neural network to finalize the segmentation
    
    Parameters
    ----------
    
    mask: numpy array
      binary mask indicating the presence of particles
    coords: numpy array
      coordinates of the particles center
      
    Returns
    -------
    
    out: numpy array
      labelled image
    """

    # Define markers for the watershed
    markers = binary_erosion((~mask).astype('int8'), disk(1))
    for n, coord in enumerate(coords):
        markers[coord[0], coord[1]] = n + 1
    markers = label(markers)

    # Computes the distance image for the mask
    grad = gradient(mask.astype('uint8'), disk(1))
    segments = watershed(grad, markers)

    # Associate each segment to the corresponding coordinate center
    matching = {}
    for n, coord in enumerate(coords):
        lb = segments[coord[0], coord[1]]
        matching[lb] = n+1

    nx, ny = segments.shape
    for x in range(nx):
        for y in range(ny):
            lb = segments[x, y]
            if(lb in matching.keys()):
                segments[x, y] = matching[lb]
            else:
                segments[x, y] = 0

    return segments

	
# -------------------------------------------------------------------
# Run the detection
# -------------------------------------------------------------------
if __name__ == '__main__':

    # Paths
    root_dir = './dataset'
    test_dir = 'imgs'

    # Select a device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parameters
    d = 7
    w = 24
    dict_weights = './weights/final.pth'
    batch_size = 1

    # Test set
    test_dataset = TestSegmentationDataset(
       root_dir= root_dir,
       input_dir= test_dir)

    testloader = torch.utils.data.DataLoader(
       test_dataset,
       batch_size = batch_size,
       num_workers = 0)

    # Constructs the neural network
    net = Net(d, w)
    net.to(device)
    net.load_state_dict(torch.load(dict_weights, map_location='cpu'))

    # Loss criterion
    criterion = nn.MSELoss()

    for i, sample in enumerate(testloader):
                
        input_img = sample['input'].to(device)
        outputs = net(input_img)
        mask, coords = extract_coordinates(outputs)
        segments = apply_watershed(mask, coords)
        np.save(sample['name'][0] + '_segments.npy', segments)

        # Display
        img = (255*input_img.numpy()[0]).astype('int')
        img = np.transpose(img, (2, 1, 0))
        fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
        ax[0].imshow(img)
        ax[0].scatter(coords[:, 1], coords[:, 0])
        ax[1].imshow(segments)
        for a in ax.ravel():
            a.set_axis_off()
        plt.tight_layout()
        plt.show()

        

