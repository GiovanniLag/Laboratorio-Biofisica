from readlif.reader import LifFile
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import cv2 as cv

# import K-means from sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


def get_clusters(image, n_clusters, image_treshold=60, N_small=10, N_big=25, N_sparse=30, brightness_threshold=0.05):
    """
    Function to get the cluster centers of the image
    :param image: image to be clustered
    :param n_clusters: number of clusters to be created
    :param image_treshold: treshold for binarization of the image
    :param N_small: number of smallest clusters to remove
    :param N_big: number of biggest clusters to remove
    :param N_sparse: number of most sparse clusters to remove after the first removal
    :param brightness_threshold: brightness threshold for the clusters (0 to 1)
    :return: cluster centers and labels
    """

    img_array = np.array(image)
    _, img_thresh = cv.threshold(img_array, 60, 255, cv.THRESH_BINARY + cv.THRESH_OTSU) # threshold the image
    img_thresh = img_thresh / 255 # make value from 0 to 1

    # get the coordinates of the pixels
    x, y = np.where(img_thresh == 1)
    coords = np.column_stack([x, y])

    # fit the KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(coords)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Identify the smallest and biggest clusters
    sorted_indices = np.argsort(labels)
    smallest_clusters_indices = sorted_indices[:N_small]
    biggest_clusters_indices = sorted_indices[-N_big:]

    # Remove the N smallest and N biggest clusters
    intermediate_points_to_keep = np.logical_and(~np.isin(labels, smallest_clusters_indices),
                                                 ~np.isin(labels, biggest_clusters_indices))
    filtered_points = coords[intermediate_points_to_keep]
    filtered_labels = labels[intermediate_points_to_keep]

    # Remove the points with brightness above the threshold
    brightness = img_array[filtered_points[:, 0], filtered_points[:, 1]]
    brightness_normalized = brightness / 255
    #for each cluster, calculate the average brightness of the pixels in the cluster
    average_brightness = []
    for i in range(n_clusters - N_small - N_big):
        cluster_points = filtered_points[filtered_labels == i]
        cluster_brightness = brightness_normalized[filtered_labels == i]
        average_brightness.append(np.mean(cluster_brightness))
    #get the indices of the clusters with brightness above the threshold
    bright_clusters_indices = np.where(np.array(average_brightness) > brightness_threshold)[0]
    #filter out the clusters with brightness above the threshold
    final_points_to_keep = ~np.isin(filtered_labels, bright_clusters_indices)
    filtered_points = filtered_points[final_points_to_keep]

    # Re-clustering the filtered points
    kmeans_filtered = KMeans(n_clusters=len(np.unique(labels)) - N_small - N_big).fit(filtered_points)
    filtered_cluster_labels = kmeans_filtered.labels_
    filtered_centroids = kmeans_filtered.cluster_centers_

    # Calculate sparsity for the new clusters
    average_distances = []
    for i, centroid in enumerate(filtered_centroids):
        cluster_points = filtered_points[filtered_cluster_labels == i]
        distances = pairwise_distances(cluster_points, [centroid])
        average_distance = np.mean(distances)
        average_distances.append(average_distance)

    # Determine the N most sparse clusters
    sparse_clusters_indices = np.argsort(average_distances)[-N_sparse:]

    # Filter out the N most sparse clusters
    final_points_to_keep = ~np.isin(filtered_cluster_labels, sparse_clusters_indices)
    final_filtered_points = filtered_points[final_points_to_keep]

    # Re-run K-means clustering
    kmeans_final = KMeans(n_clusters=200).fit(final_filtered_points)
    final_cluster_centers = kmeans_final.cluster_centers_
    final_cluster_labels = kmeans_final.labels_

    return final_cluster_centers, final_cluster_labels, final_filtered_points