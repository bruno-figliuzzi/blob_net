"""
Compute segmentation metrics for the experimental with an available ground 
truth
"""

import os
from math import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from skimage.io import imread
from skimage.measure import regionprops, label
from skimage.segmentation import mark_boundaries
from sklearn.metrics.pairwise import euclidean_distances
 

def compute_IoU(img1, img2, lb1, lb2):

    """
    Computes the intersection over union between the object with label lb1 
    in the first image and the object with label lb2 in the second image

    Parameters
    ----------

    img1: numpy array
       first image
    img2: numpy array
       second image (with the same size as img1)
    lb1: int
       label of the object in the first image
    lb2: int
       label of the object in the second image
    """

    mask1 = (img1 == lb1)
    mask2 = (img2 == lb2)
    U = np.sum(np.maximum(mask1, mask2)) + 1e-12
    I = np.sum(np.minimum(mask1, mask2))
    return I/U



def compute_metrics(img, img_truth, IoU_threshold):

    """
    Computes segmentation metrics by comparing the output of the segmentation
    to a groundtruth segmentation

    Parameters
    ----------

    img: numpy array
       segmentation result
    img_truth: numpy array
       corresponding ground truth image
    IoU_threshold: float
       IoU threshold for a proper detection

    Returns
    -------

    precision: float
       segmentation precision
    recall: float
       segmentation recall
    """

    # Region properties for the segmented/ground truth images
    props = regionprops(img)
    props_truth = regionprops(img_truth)
    
    # Number of particles/detections
    n_particles = np.max(img_truth)
    n_detections = np.max(img)

    
    centroids = np.array([np.array(prop.centroid) for prop in props])
    centroids_truth = np.array([np.array(prop.centroid) for 
      prop in props_truth])
    radii = np.array([prop.equivalent_diameter/2. for prop in props]) 
    radii_truth = np.array([prop.equivalent_diameter/2. for 
      prop in props_truth])
      
    dists = euclidean_distances(centroids_truth, centroids)
    bounds = radii_truth[:, np.newaxis] + radii[np.newaxis, :]
    mask = (dists > bounds)
    dists[mask] = 1e3
    
    truth_ind, detection_ind = linear_sum_assignment(dists)
    
    # Recall metrics            
    recall_metrics = []
    img_recall = np.zeros_like(img)
    for lb in range(n_particles):
    
        if(lb in truth_ind):
        
            idx = np.where(truth_ind==lb)[0]
            lb_detection = detection_ind[idx]
            rt = radii_truth[lb]
            r = radii[lb_detection]
            IoU =  compute_IoU(img_truth, img, lb+1, lb_detection+1)
            detected = (IoU > IoU_threshold)
            
            recall_metrics.append({
                 'center': props_truth[lb].centroid,
                 'dist': float(dists[lb, lb_detection]),
                 'label': props_truth[lb].label,
                 'radii': np.linalg.norm(r - rt),
                 'detection': int(lb_detection)+1, 
                 'IoU': IoU,
                 'detected': detected})
                 
            if(detected):   
                img_recall[img_truth == lb+1] = 1
            else:
                img_recall[img_truth == lb+1] = 2            
        else:
            recall_metrics.append({'detected': False})
            img_recall[img_truth == lb+1] = 2
            
    # Precision metrics            
    precision_metrics = []
    img_precision = np.zeros_like(img)
    for lb_detection in range(n_detections):
    
        if(lb_detection in detection_ind):
        
            idx = np.where(detection_ind==lb_detection)[0]
            lb = truth_ind[idx]
            IoU =  compute_IoU(img_truth, img, lb+1, lb_detection+1)
            detected = (IoU > IoU_threshold)
            precision_metrics.append({'detected': detected})
                 
            if(detected):   
                img_precision[img == lb+1] = 1
            else:
                img_precision[img == lb_detection+1] = 2
            
        else:
            precision_metrics.append({'detected': False})
            img_precision[img == lb_detection+1] = 2
                     
    
    return recall_metrics, precision_metrics, img_recall, img_precision


  
def display_results(img, img_truth, original, img_recall, img_precision):

    img_results = np.copy(img_recall)
    img_results[img_precision==2] = 3
        
    fig, ax = plt.subplots(2,2, figsize=(10, 5), sharex=True, sharey=True)
    ax[0, 0].set_title("Segmentation")
    ax[0, 0].imshow(img)
    ax[0, 1].set_title("Ground truth")
    ax[0, 1].imshow(img_truth)
    ax[1, 0].set_title("Original")
    ax[1, 0].imshow(original)
    ax[1, 1].set_title("Results")
    ax[1, 1].imshow(img_results)
    for a in ax.ravel():
        a.set_axis_off()
    plt.tight_layout()
    plt.show()
    plt.close()   
      	
# -------------------------------------------------------------------
# Run the detection
# -------------------------------------------------------------------
if __name__ == '__main__':

    # Paths
    img_dir = './dataset/imgs/'
    segmentation_dir = './results/'
    groundtruth_dir = './dataset/img_truths/'
    crop = 1
    IoU_threshold = 1e-3
    display=False
    save_output = True
    
    # Process all images
    img_names = os.listdir(segmentation_dir)
    for img_name in img_names:

        print('Processing image ' + img_name)
        truth_name = 'img' + img_name[4:8] + '.npy'

        original = imread(img_dir + img_name[:8] + '.png')
        img = label(np.load(segmentation_dir + img_name))-1
        img_truth = label(np.load(groundtruth_dir + truth_name))-1

        recall_metrics, precision_metrics, img_recall, img_precision =\
         compute_metrics(img, img_truth, IoU_threshold)
 
        # Compute recall, precision, etc.        
        tp = len([item for item in recall_metrics if item['detected']])
        fn = len([item for item in recall_metrics if not item['detected']])
        fp = len([item for item in precision_metrics if not item['detected']])
        print('Particles: ' + str(tp + fn))
        print('Detections: ' + str(tp + fp))
        print('Recall: ' + str(tp/(tp + fn + 1e-10)))
        print('Precision: ' + str(tp/(tp + fp + 1e-10)))

        radii = np.array([item['radii'] for item in recall_metrics
         if item['detected']])
        print('Radii error: ' + str(np.mean(radii)))
        centroids = np.array([item['dist'] for item in recall_metrics
         if item['detected']])
        print('Centroid error: ' + str(np.mean(centroids)))
        IoUs = np.array([item['IoU'] for item in recall_metrics 
         if item['detected']])
        print('Average IoU: ' + str(np.mean(IoUs)))
        print(' ')
        
        if(display):
            display_results(img, img_truth, original, img_recall, img_precision)
            
        if(save_output):
        
            img_results = np.copy(img_recall)
            img_results[img_precision==2] = 3
            print(img_results.shape)
        
            fig, ax = plt.subplots(figsize=(4,3))
            ax.imshow(img_results[400:500,300:400])
            ax.set_axis_off()
            plt.tight_layout()
            plt.savefig('./results/' + img_name[:-4] + '_result.png')
            plt.close()
            
            fig, ax = plt.subplots(figsize=(4,3))
            ax.imshow(original[400:500,300:400], cmap='gray')
            ax.set_axis_off()
            plt.tight_layout()
            plt.savefig('./results/' + img_name[:-4] + '.png')
            plt.close() 
            
      
      
