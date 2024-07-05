from readlif.reader import LifFile
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import cv2 as cv

from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances

import numpy as np
from scipy.optimize import curve_fit


def get_clusters_bgmm(image, n_components=50, n_init=3, image_threshold=60, N_small=10, N_big=25, N_not_circular=30):
    """
    Function to get the cluster centers of the image
    :param image: image to be clustered
    :param n_components: number of clusters to be created
    :param image_threshold: threshold for binarization of the image
    :param N_small: number of smallest clusters to remove
    :param N_big: number of biggest clusters to remove
    :param N_not_circular: number of least circular clusters to remove
    :return: cluster centers and labels
    """

    img_array = np.array(image)
    _, img_thresh = cv.threshold(img_array, image_threshold, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)  # threshold the image
    img_thresh = img_thresh / 255  # make value from 0 to 1

    # Get the coordinates of the pixels
    x, y = np.where(img_thresh == 1)
    coords = np.column_stack([x, y])

    # Fit the BGMM model
    bgmm = GaussianMixture(n_components=n_components, n_init=n_init).fit(coords)
    centers = bgmm.means_
    labels = bgmm.predict(coords)

    # Count the number of pixels in each cluster
    cluster_sizes = np.bincount(labels, minlength=n_components)

    # Identify the smallest and largest clusters by size
    smallest_clusters = np.argsort(cluster_sizes)[:N_small]
    largest_clusters = np.argsort(cluster_sizes)[-N_big:]

    # Calculate circularity for each cluster
    covariance_matrices = bgmm.covariances_
    determinants = np.linalg.det(covariance_matrices) # Determinatns are the generalized variance of the clusters
    traces = np.trace(covariance_matrices, axis1=1, axis2=2)
    circularity_ratios = determinants / traces
    # Identify the least circular clusters
    not_circular_clusters = np.argsort(circularity_ratios)[-N_not_circular:]

    # Combine the clusters to be removed
    clusters_to_remove = np.unique(np.concatenate([smallest_clusters, largest_clusters, not_circular_clusters]))

    # Create a mask for points not in the clusters to be removed
    mask_to_keep = ~np.isin(labels, clusters_to_remove)
    filtered_coords = coords[mask_to_keep]
    filtered_labels = labels[mask_to_keep]

    # Adjust the cluster centers based on remaining labels (optional, could be omitted if centers are not needed post-filtering)
    unique_labels = np.unique(filtered_labels)
    filtered_centers = centers[unique_labels]
    filtered_genralized_variances = determinants[unique_labels]

    return filtered_centers, filtered_labels, filtered_coords, filtered_genralized_variances


def circular_gaussian_2d(coords, A, x0, y0, sigma):
    """
    2D Gaussian function with circular symmetry.
    """
    x, y = coords
    return A * np.exp(-(((x-x0)**2 + (y-y0)**2) / (2*sigma**2)))

def fit_2D_gaussian(image, centers, initial_sigmas=1, max_radius=10):
    """
    Function to fit a 2D Gaussian to an image, considering only pixels within a specified max_radius from the centers,
    and handling failed fits by excluding them from the output.
    
    :param image: Image (2D array) to be fitted.
    :param centers: List or array of (x, y) tuples for the centers of the Gaussians.
    :param initial_sigmas: Initial value for sigma.
    :param max_radius: Maximum radius from center to include pixels in the fit.
    :return: A list of fitted parameters for each Gaussian and a list of failed instances. fitted_params is composed of: [amplitude, x0, y0, sigma].
    """
    
    fitted_params = []
    failed_instances = []

    if np.isscalar(initial_sigmas):
        initial_sigmas = np.full(len(centers), initial_sigmas)

    # Convert image to NumPy array if it's not already one
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    for i, center in enumerate(centers):
        x0, y0 = center
        
        # Create a meshgrid for the x and y coordinates
        x = np.arange(0, image.shape[1], 1)
        y = np.arange(0, image.shape[0], 1)
        x, y = np.meshgrid(x, y)
        
        # Calculate the distance of all points to the center
        distances = np.sqrt((x - x0)**2 + (y - y0)**2)
        
        # Select only points within the max_radius
        within_radius = distances <= max_radius
        x_within_radius = x[within_radius]
        y_within_radius = y[within_radius]
        image_within_radius = image[within_radius]
        
        # Flatten the x, y, and image arrays for points within max_radius
        x_flat = x_within_radius.flatten()
        y_flat = y_within_radius.flatten()
        image_flat = image_within_radius.flatten()
        
        # Initial parameter guess: [amplitude, x0, y0, sigma]
        initial_guess = [np.max(image_within_radius), x0, y0, initial_sigmas[i]]
        
        # Attempt to use curve_fit to fit the Gaussian, with the simplified model
        try:
            popt, _ = curve_fit(circular_gaussian_2d, (x_flat, y_flat), image_flat, p0=initial_guess, bounds=(0, np.inf))
            fitted_params.append(popt)
        except RuntimeError:
            failed_instances.append(center)
    
    return fitted_params, failed_instances

def FWHM(sigma):
    """
    Function to calculate the full width at half maximum (FWHM) of a Gaussian distribution.
    
    :param sigma: Standard deviation of the Gaussian distribution.
    :return: FWHM of the Gaussian distribution.
    """
    return 2 * np.sqrt(2 * np.log(2)) * sigma