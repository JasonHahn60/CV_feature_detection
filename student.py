import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops
from scipy.ndimage import convolve, gaussian_filter
from skimage.feature import peak_local_max
from skimage.filters import sobel_h, sobel_v
from skimage.transform import resize, rotate
from skimage.feature import hog
import scipy


def plot_interest_points(image, x, y):
    '''
    Plot interest points for the input image. 
    
    Show the interest points given on the input image. Be sure to add the images you make to your writeup. 

    Useful functions: Some helpful (not necessarily required) functions may include
        - matplotlib.pyplot.imshow, matplotlib.pyplot.scatter, matplotlib.pyplot.show, matplotlib.pyplot.savefig
    
    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    '''

    # TODO: Your implementation here! See block comments and the handout pdf for instructions
    plt.imshow(image, cmap='gray')
    plt.scatter(x, y, c='red', marker='o', s = 15)
    plt.show()

def get_interest_points(image, feature_width):
    # STEP 1: Calculate the gradient (partial derivatives on two directions).
    # STEP 2: Apply Gaussian filter with appropriate sigma.
    # STEP 3: Calculate Harris cornerness score for all pixels.
    # STEP 4: Peak local max to eliminate clusters. (Try different parameters.)
    
    # BONUS: There are some ways to improve:
    # 1. Making interest point detection multi-scaled.
    # 2. Use adaptive non-maximum suppression.

    #Compute derivatives
    Ix = convolve(image, np.array([[-1, 0, 1]]))
    Iy = convolve(image, np.array([[-1], [0], [1]]))

    #Compute elements of M matrix
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    IxIy = Ix * Iy

    #Compute Harris response
    window = np.ones((feature_width, feature_width))
    Sxx = convolve(Ix2, window)
    Syy = convolve(Iy2, window)
    Sxy = convolve(IxIy, window)
    det_M = Sxx * Syy - Sxy ** 2
    trace_M = Sxx + Syy
    R = det_M - 0.05 * trace_M ** 2 

    #Thresholding
    threshold = 0.01 * np.max(R) 
    R[R < threshold] = 0

    #Non-maximal suppression
    corners = peak_local_max(R, min_distance=feature_width//2, threshold_abs=1e-10)

    xs = corners[:, 1]
    ys = corners[:, 0]

    return xs, ys
    

def get_features(image, x, y, feature_width):
    descriptors = []
    for i in range(len(x)):
        xi, yi = int(x[i]), int(y[i])
        if xi - feature_width // 2 < 0 or yi - feature_width // 2 < 0 or xi + feature_width // 2 > image.shape[1] or yi + feature_width // 2 > image.shape[0]:
            continue  
        
        patch = image[yi - feature_width // 2:yi + feature_width // 2, xi - feature_width // 2:xi + feature_width // 2]
        
        #gradients
        dx = np.gradient(patch, axis=1)
        dy = np.gradient(patch, axis=0)
        
        #gradient magnitude and orientation
        magnitude = np.sqrt(dx**2 + dy**2)
        orientation = np.arctan2(dy, dx) * (180 / np.pi) % 360
        
        #histogram
        descriptor = np.zeros((4, 4, 8))  #4x4 grid with 8 bins
        
        cell_size = feature_width // 4
        for row in range(4):
            for col in range(4):
                cell_magnitude = magnitude[row*cell_size:(row+1)*cell_size, col*cell_size:(col+1)*cell_size]
                cell_orientation = orientation[row*cell_size:(row+1)*cell_size, col*cell_size:(col+1)*cell_size]
                
                for bin in range(8):
                    ori_min = (bin * 45) % 360
                    ori_max = ((bin + 1) * 45) % 360
                    
                    #accummulate into bins
                    bin_mask = (cell_orientation >= ori_min) & (cell_orientation < ori_max)
                    descriptor[row, col, bin] = np.sum(cell_magnitude[bin_mask])
        
        descriptor = descriptor.flatten()
        descriptor /= (np.linalg.norm(descriptor) + 1e-10) 
        
        descriptors.append(descriptor)

    return np.array(descriptors)

    
def match_features(im1_features, im2_features):
    distances = scipy.spatial.distance.cdist(im1_features, im2_features, 'euclidean')
    idx_closest = np.argsort(distances, axis=1)[:, :2]
    ratios = distances[np.arange(distances.shape[0]), idx_closest[:, 0]] / distances[np.arange(distances.shape[0]), idx_closest[:, 1]]

    #Dynamic NNDR
    threshold = np.percentile(ratios, 70) 

    matches_mask = ratios < threshold
    matches = np.column_stack((np.where(matches_mask)[0], idx_closest[matches_mask, 0]))
    confidences = 1 - ratios[matches_mask]
    sorted_indices = np.argsort(-confidences)
    matches = matches[sorted_indices]
    confidences = confidences[sorted_indices]

    return matches, confidences

