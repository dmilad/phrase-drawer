"""
Edge Orientation Detection

This module provides functionality for detecting edge orientations in images,
which can be used to align overlaid text/images with local edge directions.
"""
import numpy as np
from PIL import Image
from scipy import ndimage


def compute_structure_tensor_orientation(img_array, smoothing_sigma=2.0, integration_sigma=2.0):
    """
    Compute orientation using structure tensor method.

    The structure tensor method is more robust than simple gradient-based methods
    as it considers the local neighborhood structure and provides coherence weighting.

    Args:
        img_array: Grayscale image array
        smoothing_sigma: Standard deviation for Gaussian smoothing before edge detection
        integration_sigma: Standard deviation for Gaussian smoothing of structure tensor components

    Returns:
        tuple: (orientation, magnitude)
            - orientation: Array of angles in radians
            - magnitude: Array of orientation confidence/strength values
    """
    # Apply Gaussian smoothing
    if smoothing_sigma > 0:
        img_array = ndimage.gaussian_filter(img_array, sigma=smoothing_sigma)

    # Compute gradients
    gradient_x = ndimage.sobel(img_array, axis=1)
    gradient_y = ndimage.sobel(img_array, axis=0)

    # Structure tensor components
    Ixx = ndimage.gaussian_filter(gradient_x * gradient_x, sigma=integration_sigma)
    Iyy = ndimage.gaussian_filter(gradient_y * gradient_y, sigma=integration_sigma)
    Ixy = ndimage.gaussian_filter(gradient_x * gradient_y, sigma=integration_sigma)

    # Compute orientation from structure tensor
    # Dominant orientation is given by eigenvector of structure tensor
    orientation = 0.5 * np.arctan2(2 * Ixy, Ixx - Iyy)

    # Compute coherence (measure of orientation strength)
    # Coherence = (λ1 - λ2) / (λ1 + λ2), where λ1, λ2 are eigenvalues
    trace = Ixx + Iyy
    det = Ixx * Iyy - Ixy * Ixy
    lambda1 = 0.5 * (trace + np.sqrt(trace**2 - 4*det + 1e-10))
    lambda2 = 0.5 * (trace - np.sqrt(trace**2 - 4*det + 1e-10))
    coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-10)

    # Use coherence as magnitude (orientation confidence)
    magnitude = coherence * np.sqrt(Ixx + Iyy)

    return orientation, magnitude


def compute_sobel_orientation(img_array, smoothing_sigma=2.0):
    """
    Compute orientation using Sobel operator method.

    This is the original method that uses Sobel gradients to compute edge orientations.

    Args:
        img_array: Grayscale image array
        smoothing_sigma: Standard deviation for Gaussian smoothing before edge detection

    Returns:
        tuple: (orientation, magnitude)
            - orientation: Array of angles in radians
            - magnitude: Array of gradient magnitudes
    """
    # Apply Gaussian smoothing
    if smoothing_sigma > 0:
        img_array = ndimage.gaussian_filter(img_array, sigma=smoothing_sigma)

    # Compute gradients using Sobel
    gradient_x = ndimage.sobel(img_array, axis=1)
    gradient_y = ndimage.sobel(img_array, axis=0)

    # Compute magnitude and orientation
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Compute gradient orientation
    # The gradient points perpendicular to edges, so we add 90° (π/2 radians)
    # to get the orientation along edges (parallel to them)
    gradient_orientation = np.arctan2(gradient_y, gradient_x)

    # Add 90° to align parallel to edges instead of perpendicular
    orientation = gradient_orientation + np.pi/2

    return orientation, magnitude


def compute_orientation_field(image_path, chunk_size=5, smoothing_sigma=2.0, method='structure_tensor', integration_sigma=2.0):
    """
    Compute the orientation field of an image based on edge directions.

    The orientation represents the direction along which edges run (not perpendicular to them).
    For example:
    - 0° means horizontal edges (left-right)
    - 45° means diagonal edges (bottom-left to top-right)
    - 90° means vertical edges (bottom-top)
    - -45° means diagonal edges (top-left to bottom-right)

    Args:
        image_path: Path to the source image
        chunk_size: Size of chunks to divide the image into (default: 5)
        smoothing_sigma: Standard deviation for Gaussian smoothing of gradients (default: 2.0)
                        Higher values create smoother, more coherent orientation fields
        method: Orientation computation method (default: 'structure_tensor')
                Options: 'structure_tensor', 'sobel'
        integration_sigma: Standard deviation for structure tensor integration (default: 2.0)
                          Only used when method='structure_tensor'

    Returns:
        tuple: (orientations, chunk_centers)
            - orientations: numpy array of angles in degrees for each chunk
            - chunk_centers: list of (x, y) tuples for chunk centers
    """
    # Load image and convert to grayscale
    img = Image.open(image_path)

    # Handle transparency by compositing onto white background
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        # Create a white background
        background = Image.new('RGB', img.size, (255, 255, 255))
        # Paste the image on the background using the alpha channel as mask
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
        img = background

    # Now convert to grayscale
    img = img.convert('L')
    img_array = np.array(img, dtype=float)

    # Compute orientations based on selected method
    if method == 'structure_tensor':
        edge_orientation, gradient_magnitude = compute_structure_tensor_orientation(
            img_array, smoothing_sigma, integration_sigma
        )
    elif method == 'sobel':
        edge_orientation, gradient_magnitude = compute_sobel_orientation(
            img_array, smoothing_sigma
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'structure_tensor' or 'sobel'.")

    # Divide image into chunks and compute average orientation per chunk
    height, width = img_array.shape
    chunk_centers = []
    orientations = []

    for y in range(0, height, chunk_size):
        for x in range(0, width, chunk_size):
            # Define chunk boundaries
            y_end = min(y + chunk_size, height)
            x_end = min(x + chunk_size, width)

            # Extract chunk data
            chunk_orientations = edge_orientation[y:y_end, x:x_end]
            chunk_magnitudes = gradient_magnitude[y:y_end, x:x_end]

            # Compute weighted average orientation using gradient magnitude as weight
            # This gives more importance to strong edges
            total_magnitude = np.sum(chunk_magnitudes)

            if total_magnitude > 0:
                # Use weighted circular mean for angles
                # Convert to complex numbers to handle angle wraparound properly
                complex_orientations = np.exp(1j * chunk_orientations)
                weighted_sum = np.sum(complex_orientations * chunk_magnitudes)
                avg_orientation = np.angle(weighted_sum)
            else:
                # If no strong edges, default to 0
                avg_orientation = 0

            # Convert to degrees
            orientation_degrees = np.degrees(avg_orientation)

            # Calculate chunk center
            center_x = x + (x_end - x) // 2
            center_y = y + (y_end - y) // 2

            chunk_centers.append((center_x, center_y))
            orientations.append(orientation_degrees)

    return np.array(orientations), chunk_centers
