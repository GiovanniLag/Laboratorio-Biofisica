import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def circular_gaussian_2d(coords, A, x0, y0, sigma):
    """
    2D Gaussian function with circular symmetry.
    """
    x, y = coords
    return A * np.exp(-(((x-x0)**2 + (y-y0)**2) / (2 * sigma**2)))

def plot_3d_gaussian_fit(image, fitted_params, max_radius=10):
    """
    Plots the original data and the fitted 2D Gaussian functions for multiple centers.
    """
    
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    fig = plt.figure(figsize=(12, 6))

    # Original Data Plot
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(0, image.shape[1])
    y = np.arange(0, image.shape[0])
    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, image, cmap='viridis', edgecolor='none', alpha=0.5)
    ax.set_title('Original Data & Fitted Gaussians')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Intensity')

    # Plotting each Gaussian
    for params in fitted_params:
        A, x0, y0, sigma = params
        distances = np.sqrt((x - x0)**2 + (y - y0)**2)
        mask = distances <= max_radius
        z = np.zeros_like(image)
        z[mask] = circular_gaussian_2d((x[mask], y[mask]), A, x0, y0, sigma)
        # Use wireframe or plot_surface with different colors for clarity
        ax.plot_surface(x, y, z, rstride=1, cstride=1, color='r', alpha=0.5, linewidth=0, antialiased=False)

    plt.tight_layout()
    plt.show()